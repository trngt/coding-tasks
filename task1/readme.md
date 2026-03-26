# Draft Response:

### Notes:
Strengths:
Currently, the project has a strong framework for the prototyping workflow for training and prediction: data loading, model setup, training, and output.

Issues:
The project does not have functionality for easy comparisons, between architectures, data, and ablation. The project will need a way to configure these different model types, switch between them, and compare their different output results.
- Fix configuration:
	- Currently the model is defined in the class and loaded in the trainer.
- Data is loaded from a csv
	- For collaboration, a single data source will be optimal.
- Output is fixed to hard-coded paths.

Proposed solution:
- Define a configuration class:
	- Model architectures, ablation, attention mechanisms are defined in sub-transformer classes
		- And configuration loads appropriate transformer class by name: (e.g. NaiveProteinTransformer, EightHeadProteinTransformer, AblatedHeadProteinTransformer)
	- Dataset configuration (* see below)
	- Fitness measurements, encode in dataset (** determine if fitness is a data level configuration or a training one.)
	- Name (for file saving)

- Create a unified data storage location for collaboration
	- Create a data loading procedure that downloads the dataset

- Output
	- Create a file saving naming convention which includes the name of the configuration and the run (datetime and the user who ran the model, and a unique hash if necessary)
	- Potentially these models will run on a computing cluster and save to a unified output space, for which collaborators can compare model runs.
	- An overkill solution if collaboration is needed for more complex runs:
		A SQL database can be a longterm solution if more comprehensivee metadata is needed per run.
		(git commit hash for model run, hash of dataset, etc...)

Open questions:
- Verify that ablation and attention mechanisms can be customized in Transformer subclasses.
- Work through potential output format more carefully, is a SQL database necessary? 
- What does collaboration look like? Those who can code and write custom subclasses? Or do we stick to configuration files?

