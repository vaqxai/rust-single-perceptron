use std::{io, env};

use perceptron::Perceptron;

mod fileread;
mod perceptron;
mod math;
mod tests;

fn get_inp() -> String {
	let mut input = String::new();
	io::stdin().read_line(&mut input).unwrap();
	let input = input.trim();
	input.to_owned()
}

fn get_inp_vec() -> Vec<f64> {
	println!("Please enter your test Vector:");
	let input = get_inp();
	let input = match input.split(" ").map(|x| x.parse::<f64>())
		.collect::<Result<Vec<f64>, _>>() {
		Ok(x) => x,
		Err(_) => {
			panic!("Invalid input");
		}
	};
	return input;
}

fn get_learning_rate_from_user() -> f64 {
	println!("Please enter training rate (0.0 - 1.0):");
	let input = match get_inp().parse::<f64>() {
		Ok(x) if (x >= 0.0 && x <= 1.0) => x,
		Ok(_) => panic!("Please input a value between 0.0 and 1.0"),
		Err(_) => panic!("Invalid input")
	};
	return input;
}

fn main() {
	let args: Vec<String> = env::args().collect();
	let command;
	if args.len() > 1 {
		command = args[1].as_str();
	} else {
		println!("Please enter a valid command (iris, ex1, ex1_enterdata, iris_enterdata)");
		return;
	}

	match command {
		"iris" => {

			let learning_rate = get_learning_rate_from_user();

			let mut p = train_on_labeled("ip1/training.txt","Iris-virginica", "Iris-versicolor", learning_rate);
			let efficiency = test_on_labeled(&mut p, "ip1/test.txt","Iris-virginica", "Iris-versicolor");

			println!("Efficiency: {}%", efficiency * 100.0);

			return;

		},
		"ex1" => {

			let learning_rate = get_learning_rate_from_user();

			let mut p = train("ex1/train.txt", learning_rate);
			let efficiency = test_on_file(&mut p, "ex1/test.txt");

			println!("Efficiency: {}%", efficiency * 100.0);

			return;

		},
		"ex1_enterdata" => {

			let learning_rate = get_learning_rate_from_user();

			let p = train("ex1/train.txt", learning_rate);

			let input = get_inp_vec();
			let outcome = p.predict(&input);

			println!("The perceptron predicted: {}", outcome);

			return;
		}
		"iris_enterdata" => {

			let learning_rate = get_learning_rate_from_user();

			let p = train_on_labeled("ip1/training.txt", "Iris-virginica", "Iris-versicolor", learning_rate);

			let input = get_inp_vec();

			let outcome = p.predict(&input);

			if outcome == 0.0 {
				println!("The perceptron predicted: Iris-versicolor");
			} else {
				println!("The perceptron predicted: Iris-virginica");
			}

			return;

		},
		&_ => {
			println!("Please enter a valid command (iris, ex1, ex1_enterdata, iris_enterdata)");
		}
	}

}

fn train_on_labeled(data_path: &str, raised_label: &str, lowered_label: &str, learning_rate: f64) -> Perceptron {
	let mut lines = fileread::read_file(data_path).unwrap();

	let dimensions = match lines.next() {
		Some(line) => line.unwrap().split(",").count()-1,
		None => panic!("Error reading file")
	};

	let mut p = Perceptron::new(dimensions, learning_rate);
	let lines = fileread::read_file(data_path).unwrap();
	let mut i = 1;
	for line in lines {
		let line = line.unwrap();
		let count = line.split(",").count();

		if count < dimensions {
			println!("Malformed line: {}, skipping.", i);
			continue;
		}

		let cols : Vec<&str> = line.split(",").collect() ;
		let inputs: Vec<f64> = cols[..cols.len()-1].iter().map(|x| x.parse::<f64>().unwrap()).collect();
		let determinant = line.split(",").last().unwrap();
		let expected_output = match determinant {
			_ if determinant == raised_label => 1.0,
			_ if determinant == lowered_label => 0.0,
			_ => panic!("Error reading file at: {}, determinant: {}, raised: {}, lowered: {}", line, determinant, raised_label, lowered_label)
		};
		println!("{:?} : {}", inputs.to_vec(), expected_output);
		p.train(inputs.to_vec(), expected_output);
		i += 1;
		if i % 10 == 0 {
			println!("{}", i);
		}
	}
	p
}

fn train(data_path: &str, learning_rate: f64) -> Perceptron {
	train_on_labeled(data_path, "1", "0", learning_rate)
}

fn test_on_file(p: &mut Perceptron, test_data_path: &str) -> f64 {
	test_on_labeled(p, test_data_path, "1", "0")
}

fn test_on_labeled(p: &mut Perceptron, test_data_path: &str, raised_label: &str, lowered_label: &str) -> f64 {
	let lines = fileread::read_file(test_data_path).unwrap();

	let mut correct = 0;
	let mut total = 0;

	for line in lines {
		let line = line.unwrap();
		let count = line.split(",").count();

		if count < p.inputs.len() {
			println!("Malformed line: {}, skipping.", total+1);
			continue;
		}
	
		let cols : Vec<&str> = line.split(",").collect() ;
		let inputs: Vec<f64> = cols[..cols.len()-1].iter().map(|x| x.parse::<f64>().unwrap()).collect();
		let determinant = line.split(",").last().unwrap();
		let expected_output = match determinant {
			_ if determinant == raised_label => 1.0,
			_ if determinant == lowered_label => 0.0,
			_ => panic!("Error reading file")
		};
		let output = p.predict(&inputs.to_vec());
		let inp = inputs.to_vec();
		//p.train(inputs[..inputs.len()-1].to_vec(), *expected_output);
		if output == expected_output {
			correct += 1;
		}
		total += 1;

		println!("Perceptron guessed {} for {:?}, should be {}, ({})", output, inp, expected_output, expected_output == output);

	}

	correct as f64 / total as f64
}