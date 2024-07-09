# Migration Notes

## 2.0 
- Changed to tensorflow 2.11+ (i.e. the Adam Optimizer is now accessed via tf.keras.optimizers.legacy.Adam)

- The training parametrization was improved. This caused breaking changes. Please adapt the following:
    - remove the quotation marks (i.e. stringifiying) of all numerical parameters in config_training.json

- Multistep optimization has been included. This is not yet fully integrated, however, you can use "--step" as a parameter in the command line and (as previously) load weights to continue with different settings. For that, the config_training.json is an array of the same format as before with the additional field "step" which is an integer.

