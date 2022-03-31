#[cfg(test)]
mod tests {

	use crate::{perceptron, train, test_on_file, train_on_labeled, test_on_labeled};

	#[test]
	pub fn train_perceptron_on_file(){
		train("ex1/train.txt", 0.1);
	}

	#[test]
	pub fn test_perceptron_classification(){
		let mut p = train("ex1/train.txt", 0.1);
		let efficiency = test_on_file(&mut p, "ex1/test.txt");

		println!("Efficiency: {}%", efficiency * 100.0);
	}

	#[test]
	pub fn test_perceptron_classification_on_iris(){
		let mut p = train_on_labeled("ip1/training.txt","Iris-virginica", "Iris-versicolor", 0.1);
		let efficiency = test_on_labeled(&mut p, "ip1/test.txt","Iris-virginica", "Iris-versicolor");

		println!("Efficiency: {}%", efficiency * 100.0);
	}

	#[test]
	pub fn test_new_perceptron() {
		let mut perceptron = perceptron::Perceptron::new(2, 1.0);
		perceptron.weights = vec![2.0, 1.0];
		perceptron.bias = 2.0;

		let outcome = perceptron.predict(&vec![-1.0, 1.0]);

		assert_eq!(outcome, 0.0);
	}

	#[test]
	pub fn test_perceptron_training() {
		let mut perceptron = perceptron::Perceptron::new(2, 0.5);
		perceptron.weights = vec![2.0, 1.0];
		perceptron.bias = 2.0;

		perceptron.train(vec![-1.0, 1.0], 1.0);

		perceptron.predict(&vec![-1.0, 1.0]);

		assert_eq!(perceptron.weights, vec![1.5, 1.5]);
	}

}