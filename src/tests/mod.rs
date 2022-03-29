#[cfg(test)]
mod tests {

	use crate::{perceptron, train};
	use super::*;

	#[test]
	pub fn test_perceptron_on_file(){
		train("ex1/train.txt");
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