// Vec<Vec<Vec<f32>>> -> 3D matrix => je mehr Vec desto höher die Dimension
use std::convert::TryInto;
use std::fs::File;
use std::io::Read;

struct Matrix2D {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

struct LinearLayer {
    weights: Matrix2D,
    biases: Matrix2D,
}

struct Neuron {
    layer1: LinearLayer,
    layer2: LinearLayer,
}
#[allow(dead_code)]
struct LayerCache {
    input: Matrix2D,
    z: Matrix2D,
    output: Matrix2D,
    activ_derivative: Matrix2D,
}

impl Matrix2D {
    fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }
    fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    fn relu(&self) -> Matrix2D {
        let new_data = self.data.iter().map(|&x| x.max(0.0)).collect();
        Matrix2D {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }

    fn relu_derivative(&self) -> Matrix2D {
        let new_data = self
            .data
            .iter()
            .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
            .collect();
        Matrix2D {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }

    fn sigmoid(&self) -> Matrix2D {
        let new_data = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Matrix2D {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }

    fn sigmoid_derivative(&self) -> Matrix2D {
        let new_data = self
            .data
            .iter()
            .map(|&x| {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            })
            .collect();
        Matrix2D {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }

    // fn add(&self, other: &Matrix2D) -> Matrix2D {
    //     assert_eq!(self.rows, other.rows);
    //     assert_eq!(self.cols, other.cols);

    //     let new_data = self
    //         .data
    //         .iter()
    //         .zip(other.data.iter())
    //         .map(|(&a, &b)| a + b)
    //         .collect();
    //     Matrix2D {
    //         rows: self.rows,
    //         cols: self.cols,
    //         data: new_data,
    //     }
    // }
    fn subtract(&self, other: &Matrix2D) -> Matrix2D {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let new_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        Matrix2D {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }

    fn scale(&self, factor: f32) -> Matrix2D {
        let new_data = self.data.iter().map(|&x| x * factor).collect();
        Matrix2D {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }

    fn transpose(&self) -> Matrix2D {
        let mut new_data = vec![0.0_f32; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Matrix2D {
            rows: self.cols,
            cols: self.rows,
            data: new_data,
        }
    }

    fn multiply_elementwise(&self, other: &Matrix2D) -> Matrix2D {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let new_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        Matrix2D {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }

    fn sum_cols(&self) -> Matrix2D {
        let mut new_data = vec![0.0_f32; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[i] += self.data[i * self.cols + j];
            }
        }
        Matrix2D {
            rows: self.rows,
            cols: 1,
            data: new_data,
        }
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0_f32; rows * cols],
        }
    }

    fn random(rows: usize, cols: usize) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::SystemTime;

        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as u64;

        let mut data = Vec::with_capacity(rows * cols);
        let mut state = seed;

        for _ in 0..rows * cols {
            let mut hasher = DefaultHasher::new();
            state.hash(&mut hasher);
            state = hasher.finish();
            let val = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
            data.push(val);
        }

        Self { rows, cols, data }
    }

    fn col_slice(&self, start: usize, end: usize) -> Matrix2D {
        let end = end.min(self.cols);
        let new_cols = end - start;
        let mut new_data = vec![0.0_f32; self.rows * new_cols];

        for i in 0..self.rows {
            for j in 0..new_cols {
                new_data[i * new_cols + j] = self.data[i * self.cols + (start + j)];
            }
        }
        Matrix2D {
            rows: self.rows,
            cols: new_cols,
            data: new_data,
        }
    }
    fn add_bias(&self, bias: &Matrix2D) -> Matrix2D {
        assert_eq!(self.rows, bias.rows);
        assert_eq!(bias.cols, 1);

        let mut new_data = self.data.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_data[i * self.cols + j] += bias.data[i];
            }
        }
        Matrix2D {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }
}

impl Clone for Matrix2D {
    fn clone(&self) -> Self {
        Matrix2D {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

impl LinearLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: Matrix2D::random(output_size, input_size),
            biases: Matrix2D::zeros(output_size, 1),
        }
    }
    fn forward(&self, input: &Matrix2D) -> (Matrix2D, LayerCache) {
        let z = matrix_mult_struc_vec(&self.weights, input).add_bias(&self.biases); // + self.biases
        let output = z.relu();

        let cache = LayerCache {
            input: input.clone(),
            z: z.clone(),
            output: output.clone(),
            activ_derivative: z.relu_derivative(),
        };

        (output, cache)
    }
    fn forward_sigmoid(&self, input: &Matrix2D) -> (Matrix2D, LayerCache) {
        let z = matrix_mult_struc_vec(&self.weights, input).add_bias(&self.biases);
        let output = z.sigmoid(); // ← statt relu
        let cache = LayerCache {
            input: input.clone(),
            z: z.clone(),
            output: output.clone(),
            activ_derivative: z.sigmoid_derivative(), // ← Berechnung der Sigmoid-Ableitung
        };
        (output, cache)
    }

    fn backward(
        &self,
        cache: &LayerCache,
        grad_output: &Matrix2D,
    ) -> (Matrix2D, Matrix2D, Matrix2D) {
        let m = cache.input.cols as f32;

        let dz = grad_output.multiply_elementwise(&cache.activ_derivative);
        let dw = matrix_mult_struc_vec(&dz, &cache.input.transpose()).scale(1.0 / m);
        let db = dz.sum_cols().scale(1.0 / m);
        let grad_input = matrix_mult_struc_vec(&self.weights.transpose(), &dz);

        (grad_input, dw, db)
    }
}

impl Neuron {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            layer1: LinearLayer::new(input_size, hidden_size),
            layer2: LinearLayer::new(hidden_size, output_size),
        }
    }

    fn forward(&self, input: &Matrix2D) -> (Matrix2D, LayerCache, LayerCache) {
        let (a1, cache1) = self.layer1.forward(input);
        let (a2, cache2) = self.layer2.forward_sigmoid(&a1);
        (a2, cache1, cache2)
    }

    fn calculate_loss(&self, predicted: &Matrix2D, target: &Matrix2D) -> f32 {
        let m = predicted.data.len() as f32;
        let mut total_loss = 0.0;
        let eps = 1e-7; // Vermeidung von log(0)

        for (&p, &t) in predicted.data.iter().zip(target.data.iter()) {
            let p = p.clamp(eps, 1.0 - eps);
            total_loss += -t * p.ln() - (1.0 - t) * (1.0 - p).ln();
        }
        total_loss / m
    }

    fn backward(
        &mut self,
        cache1: &LayerCache,
        cache2: &LayerCache,
        pred: &Matrix2D,
        target: &Matrix2D,
        lr: f32,
    ) {
        let grad_a2 = pred.subtract(target);
        let (grad_a1, dw2, db2) = self.layer2.backward(cache2, &grad_a2);
        let (_, dw1, db1) = self.layer1.backward(cache1, &grad_a1);

        // Gewichte und Biases aktualisieren
        self.layer2.weights = self.layer2.weights.subtract(&dw2.scale(lr));
        self.layer2.biases = self.layer2.biases.subtract(&db2.scale(lr));
        self.layer1.weights = self.layer1.weights.subtract(&dw1.scale(lr));
        self.layer1.biases = self.layer1.biases.subtract(&db1.scale(lr));
    }
    fn accuracy(&self, predicted: &Matrix2D, target: &Matrix2D) -> f32 {
        let correct = predicted
            .data
            .iter()
            .zip(target.data.iter())
            .filter(|(&p, &t)| {
                let p_class = if p > 0.5 { 1.0 } else { 0.0 };
                p_class == t
            })
            .count();

        correct as f32 / target.data.len() as f32
    }
    fn get_batches(
        &self,
        x: &Matrix2D,
        y: &Matrix2D,
        batch_size: usize,
    ) -> Vec<(Matrix2D, Matrix2D)> {
        let m = y.cols; // Anzahl Beispiele
        let mut batches = Vec::new();
        let mut i = 0;

        while i < m {
            let end = (i + batch_size).min(m);
            batches.push((x.col_slice(i, end), y.col_slice(i, end)));
            i += batch_size;
        }
        batches
    }

    fn train(&mut self, x: &Matrix2D, y: &Matrix2D, epochs: usize, lr: f32, batch_size: usize) {
        let total_batches = (y.cols + batch_size - 1) / batch_size;

        for epoch in 0..epochs {
            let batches = self.get_batches(x, y, batch_size);

            for (batch_idx, (x_batch, y_batch)) in batches.iter().enumerate() {
                print!(
                    "\rEpoch {}/{} || Batch {}/{}",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    total_batches
                );
                std::io::Write::flush(&mut std::io::stdout()).unwrap();

                let (predicted, cache1, cache2) = self.forward(x_batch);
                self.backward(&cache1, &cache2, &predicted, y_batch, lr);
            }

            let (predicted, _, _) = self.forward(x);
            let loss = self.calculate_loss(&predicted, y);
            let acc = self.accuracy(&predicted, y);
            println!(
                "\rEpoch {}/{}: Loss = {:.4}, Accuracy = {:.2}%",
                epoch + 1,
                epochs,
                loss,
                acc * 100.0
            );
        }
    }
}

fn read_picture(path: &str) -> Vec<Vec<f32>> {
    let mut file = File::open(path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let num_images = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(buffer[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(buffer[12..16].try_into().unwrap()) as usize;
    let pixels = rows * cols;

    //Bild einlesen
    let mut images = Vec::new();
    for i in 0..num_images {
        let start = 16 + i * pixels;
        let image: Vec<f32> = buffer[start..start + pixels]
            .iter()
            .map(|&p| p as f32 / 255.0)
            .collect();
        images.push(image);
    }
    images
}

fn read_labels(path: &str) -> Vec<u8> {
    let mut file = File::open(path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let num_labels = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;

    buffer[8..8 + num_labels].to_vec()
}

fn get_zero_and_ones(image_path: &str, label_path: &str) -> (Vec<Vec<f32>>, Vec<u8>) {
    let all_images = read_picture(image_path);
    let all_labels = read_labels(label_path);

    assert_eq!(all_images.len(), all_labels.len());

    let mut filterd_images = Vec::new();
    let mut filterd_labels = Vec::new();

    // Bilder und Labels filtern gleichzeitig
    for (image, label) in all_images.into_iter().zip(all_labels.into_iter()) {
        if label == 0 || label == 1 {
            filterd_images.push(image);
            filterd_labels.push(label);
        }
    }
    (filterd_images, filterd_labels)
}

fn flatten_and_transpose(image: &Vec<Vec<f32>>) -> Vec<f32> {
    let num_images = image.len();
    if num_images == 0 {
        return Vec::new();
    }
    let pixels_per_image = image[0].len();

    let mut result = Vec::with_capacity(num_images * pixels_per_image);

    for pixel_idx in 0..pixels_per_image {
        for image_idx in 0..num_images {
            result.push(image[image_idx][pixel_idx]);
        }
    }
    result
}

fn matrix_mult_struc_vec(matrix_1: &Matrix2D, matrix_2: &Matrix2D) -> Matrix2D {
    assert_eq!(matrix_1.cols, matrix_2.rows);

    let rows = matrix_1.rows;
    let cols = matrix_2.cols;
    let common_dim = matrix_1.cols;

    let mut result_data = vec![0.0_f32; rows * cols];

    for i in 0..rows {
        for k in 0..common_dim {
            // let mut sum = 0.0;
            for j in 0..cols {
                result_data[i * cols + j] += matrix_1.get(i, k) * matrix_2.get(k, j);
                // sum += matrix_1.get(i, k) * matrix_2.get(k, j);
            }
            // result_data[i * cols + j] = sum;
        }
    }
    Matrix2D {
        data: result_data,
        rows,
        cols,
    }
}

fn main() {
    let start = std::time::Instant::now();
    let (train_images, train_labels) = get_zero_and_ones(
        "../data/MNIST/raw/train-images-idx3-ubyte",
        "../data/MNIST/raw/train-labels-idx1-ubyte",
    );

    let flat_train = flatten_and_transpose(&train_images);
    let num_train = train_images.len();
    let pixels = train_images[0].len();
    let x_train = Matrix2D::new(pixels, num_train, flat_train);

    let label_data: Vec<f32> = train_labels.iter().map(|&l| l as f32).collect();
    let y_train = Matrix2D::new(1, num_train, label_data);

    let mut neuron = Neuron::new(pixels, 20, 1);
    println!("Trainingsbilder: {}", train_images.len());
    println!("X dims: {}x{}", x_train.rows, x_train.cols);
    println!("Y dims: {}x{}", y_train.rows, y_train.cols);
    neuron.train(&x_train, &y_train, 1, 0.01, 16);

    let (test_images, test_labels) = get_zero_and_ones(
        "../data/MNIST/raw/t10k-images-idx3-ubyte",
        "../data/MNIST/raw/t10k-labels-idx1-ubyte",
    );

    let flat_test = flatten_and_transpose(&test_images);
    let num_test = test_images.len();
    let x_test = Matrix2D::new(pixels, num_test, flat_test);

    let label_data_test: Vec<f32> = test_labels.iter().map(|&l| l as f32).collect();
    let y_test = Matrix2D::new(1, num_test, label_data_test);

    let (predicted, _, _) = neuron.forward(&x_test);
    let acc = neuron.accuracy(&predicted, &y_test);
    println!("Test Accuracy: {:.2}%", acc * 100.0);
    println!("Zeit: {:.2?}", start.elapsed());
}
