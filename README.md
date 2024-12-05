# Neural Network for Neutron Star Mass-Radius Prediction

This repository contains a neural network (NN) model for predicting neutron star mass-radius relationships from piecewise polytropic EoS from Read+ 2008.

## Files

- **NNmodel_GR.ckp**: Contains the weights and biases of the neural network.
- **scalerX.gz**: Transformation information for the input data.
- **scalery.gz**: Transformation information for the output data.
- **predictMR.py**: Script to use the neural network for predictions.
- **NN_run.ipynb**: Jupyter Notebook with examples of how to use the neural network.

## Data
**nsmassrad.txt** contains observed mass and radius data for inference. See [1] for reference.
All CSV files in this repository contain mass-radius data calculated using the RK4 method.

## Usage

To use the neural network for predictions, run the `predictMR.py` script. Examples of how to use the neural network can be found in the `NN_run.ipynb` notebook.

## Examples

Refer to the `NN_run.ipynb` notebook for detailed examples and usage instructions.

## License

This project is licensed under the MIT License 

## Citation
If this repository is useful for you, consider cite

[1] [in prep] 

## References
Read, J. S., Lackey, B. D., Owen, B. J., & Friedman, J. L. (2009). Constraints on a phenomenologically parametrized neutron-star equation of state. Physical Review D—Particles, Fields, Gravitation, and Cosmology, 79(12), 124032.
