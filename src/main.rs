use perceptron::Perceptron;

mod fileread;
mod perceptron;
mod math;
mod tests;

fn main() {

}

fn train_on_labeled(data_path: &str, raised_label: &str, lowered_label: &str) -> Perceptron {
	let mut lines = fileread::read_file(data_path).unwrap();

	let dimensions = match lines.next() {
		Some(line) => line.unwrap().split(",").count()-1,
		None => panic!("Error reading file")
	};

	let mut p = Perceptron::new(dimensions, 0.1);
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

fn train(data_path: &str) -> Perceptron {
	train_on_labeled(data_path, "1", "0")
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