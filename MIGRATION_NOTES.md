# Migration Notes

## 2.0 
- Changed to tensorflow 2.11+ (i.e. the Adam Optimizer is now accessed via tf.keras.optimizers.legacy.Adam)

- The training parametrization was improved. This caused breaking changes. Please adapt the following:
    - remove the quotation marks (i.e. stringifiying) of all numerical parameters in config_training.json

- Multistep optimization has been included. Additionally, json schemas are provided for all configuration files.

- Tf probability lbfgs implementation has been attached and gives an additional configuration option "optimizer"="lbfgs_tf_probability".

