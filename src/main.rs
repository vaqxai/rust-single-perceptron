use perceptron::Perceptron;

mod fileread;
mod perceptron;
mod math;
mod tests;

fn main() {

}

fn train(data_path: &str) -> Perceptron {

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
	
		let inputs: Vec<f64> = line.split(",").map(|x| x.parse::<f64>().unwrap()).collect();
		let expected_output = inputs.last().unwrap();
		println!("{:?}", inputs[..inputs.len()].to_vec());
		p.train(inputs[..inputs.len()].to_vec(), *expected_output);
		i += 1;
		if i % 10 == 0 {
			println!("{}", i);
		}
	}
	p
}