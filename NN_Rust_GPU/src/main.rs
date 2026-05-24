use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn_cuda::{Cuda, CudaDevice};
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;

type MyBackend = Autodiff<Cuda>;

// ─── Modell ─────────────────────────────────────────────────────────────────
#[derive(Module, Debug)]
struct NeuronModel<B: Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
}

#[derive(Config, Debug)]
struct NeuronConfig {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

impl NeuronConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> NeuronModel<B> {
        NeuronModel {
            layer1: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            layer2: LinearConfig::new(self.hidden_size, self.output_size).init(device),
        }
    }
}

impl<B: Backend> NeuronModel<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(x);
        let x = burn::tensor::activation::relu(x); // ReLU Layer 1
        let x = self.layer2.forward(x);
        burn::tensor::activation::sigmoid(x) // Sigmoid Layer 2
    }

    fn accuracy(&self, predicted: &Tensor<B, 2>, target: &Tensor<B, 2>) -> f32 {
        let pred_data = predicted.clone().into_data();
        let target_data = target.clone().into_data();

        let pred_vec: Vec<f32> = pred_data.to_vec().unwrap();
        let target_vec: Vec<f32> = target_data.to_vec().unwrap();

        let correct = pred_vec
            .iter()
            .zip(target_vec.iter())
            .filter(|(&p, &t)| {
                let p_class = if p > 0.5 { 1.0f32 } else { 0.0f32 };
                (p_class - t).abs() < 1e-5
            })
            .count();

        correct as f32 / target_vec.len() as f32
    }
}

impl<B: AutodiffBackend> NeuronModel<B> {
    fn forward_loss(&self, x: Tensor<B, 2>, target: Tensor<B, 2>) -> (Tensor<B, 1>, Tensor<B, 2>) {
        let predicted = self.forward(x);
        let eps = 1e-7_f32;

        // Binary Cross Entropy manuell
        let p = predicted.clone().clamp(eps, 1.0 - eps);
        let loss = target.clone() * p.clone().log()
            + (target.clone().neg() + 1.0) * (p.clone().neg() + 1.0).log();
        let loss = loss.mean().neg();

        (loss, predicted)
    }
}

// ─── Training ───────────────────────────────────────────────────────────────
fn train(
    model: NeuronModel<MyBackend>,
    x_train: Tensor<MyBackend, 2>,
    y_train: Tensor<MyBackend, 2>,
    x_test: Tensor<MyBackend, 2>,
    y_test: Tensor<MyBackend, 2>,
    epochs: usize,
    lr: f64,
    batch_size: usize,
) {
    let mut model = model;
    let mut optim = SgdConfig::new().init::<MyBackend, NeuronModel<MyBackend>>();

    let num_samples = x_train.dims()[0];
    let total_batches = (num_samples + batch_size - 1) / batch_size;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;

        for batch_idx in 0..total_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(num_samples);

            // Batch aus dem Tensor schneiden
            let x_batch = x_train.clone().slice([start..end, 0..x_train.dims()[1]]);
            let y_batch = y_train.clone().slice([start..end, 0..1]);

            let (loss, _predicted) = model.forward_loss(x_batch, y_batch);
            epoch_loss += loss.clone().into_scalar();

            // Backprop + Update
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            print!(
                "\rEpoch {}/{} || Batch {}/{}",
                epoch + 1,
                epochs,
                batch_idx + 1,
                total_batches
            );
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }

        // Evaluation auf Trainingsdaten
        let predicted = model.forward(x_train.clone());
        let acc = model.accuracy(&predicted, &y_train);
        println!(
            "\rEpoch {}/{}: Loss = {:.4}, Accuracy = {:.2}%",
            epoch + 1,
            epochs,
            epoch_loss / total_batches as f32,
            acc * 100.0
        );
    }

    // Test
    let predicted = model.forward(x_test);
    let acc = model.accuracy(&predicted, &y_test);
    println!("Test Accuracy: {:.2}%", acc * 100.0);
}

// ─── MNIST Laden ────────────────────────────────────────────────────────────
fn read_picture(path: &str) -> Vec<Vec<f32>> {
    let mut file = File::open(path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let num_images = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(buffer[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(buffer[12..16].try_into().unwrap()) as usize;
    let pixels = rows * cols;

    (0..num_images)
        .map(|i| {
            let start = 16 + i * pixels;
            buffer[start..start + pixels]
                .iter()
                .map(|&p| p as f32 / 255.0)
                .collect()
        })
        .collect()
}

fn read_labels(path: &str) -> Vec<u8> {
    let mut file = File::open(path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    let num_labels = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;
    buffer[8..8 + num_labels].to_vec()
}

fn get_zero_and_ones(image_path: &str, label_path: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    let all_images = read_picture(image_path);
    let all_labels = read_labels(label_path);

    all_images
        .into_iter()
        .zip(all_labels.into_iter())
        .filter(|(_, label)| *label == 0 || *label == 1)
        .map(|(img, label)| (img, label as f32))
        .unzip()
}

// ─── Main ───────────────────────────────────────────────────────────────────
fn main() {
    let start = std::time::Instant::now();
    let device = CudaDevice::default();

    // Daten laden
    let (train_images, train_labels) = get_zero_and_ones(
        "../data/MNIST/raw/train-images-idx3-ubyte",
        "../data/MNIST/raw/train-labels-idx1-ubyte",
    );
    let (test_images, test_labels) = get_zero_and_ones(
        "../data/MNIST/raw/t10k-images-idx3-ubyte",
        "../data/MNIST/raw/t10k-labels-idx1-ubyte",
    );

    let num_train = train_images.len();
    let num_test = test_images.len();
    let pixels = train_images[0].len();

    println!("Trainingsbilder: {}", num_train);
    println!("Testbilder: {}", num_test);

    // Flach machen: [N, 784]
    let flat_train: Vec<f32> = train_images.into_iter().flatten().collect();
    let flat_test: Vec<f32> = test_images.into_iter().flatten().collect();

    // Tensors auf GPU laden
    let x_train = Tensor::<MyBackend, 1>::from_floats(flat_train.as_slice(), &device)
        .reshape([num_train, pixels]);
    let y_train = Tensor::<MyBackend, 1>::from_floats(train_labels.as_slice(), &device)
        .reshape([num_train, 1]);

    let x_test = Tensor::<MyBackend, 1>::from_floats(flat_test.as_slice(), &device)
        .reshape([num_test, pixels]);
    let y_test =
        Tensor::<MyBackend, 1>::from_floats(test_labels.as_slice(), &device).reshape([num_test, 1]);

    // Modell initialisieren
    let model = NeuronConfig {
        input_size: pixels, // 784
        hidden_size: 20,
        output_size: 1,
    }
    .init::<MyBackend>(&device);

    train(model, x_train, y_train, x_test, y_test, 10, 0.01, 16);

    println!("Zeit: {:.2?}", start.elapsed());
}
