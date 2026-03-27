# Draft Response:

-----------

Draft text:

### The project and its strengths

This transformer-based prototype--that predicts protein fitness from sequence--is a strong foundational project, with potential for scaling up to a comprehensive research project. The code has a clear workflow with well-encapsulated components. These components create a nice scaffold for: (1) defining the model architecture, (2) loading the data, and (3) training and testing the model. For these reasons, the model is positioned well to scale up to more complex tasks, such as model comparison and collaboration.

### The requirements and the projects issues

At a high level, we are interested in extending this project in three ways: (1) allowing variation in model runs for comparison (e.g., architecture, datasets, etc.), (2) enabling project collaboration, and (3) enforcing reproducibility of model runs for publication. 

However, there are a couple of structural issues that impede these new requirements. First, the model architecture, data source, and output are all hard-coded. So, we are not able to run comparisons between architecture, datasets, fitness scores, etc. Second, the model setup, data storage, and model run output are set up to run locally. This will create organizational issues during collaboration when model runs against local datasets are run individually on each collaborator's machines. We will need a solution that streamlines collaboration and enforces reproducibility.

### Proposal solutions

To address these issues, I propose two main refactoring features: (1) model run configurations, (2) server-side data storage, and (3) version control. First, we will create a way to define model run configurations, including architecture, attention mechanisms, and data storage. Using this configuration, we can create a simple way to run the model with various configurations. The output will also need to reflect these different configurations. Second, store the data and run outputs server-side to allow for collaboration. Researchers can define various configurations and run the model server-side to compare the runs. This item is also linked to reproducibility, as a unified location for storing input and output data will create a clear, reproducible pipeline from data loading to final model run output. Finally, store the project code on a server-side code repository. This will allow collaborators to define configuration and make changes to the model on the server. As a side note, the model output should include a commit hash of the project's codebase for reproducibility.

```
Detailed requirements for reference:
	Run comparisons
	- Model comparison: ablation, architectures, attention mechanisms.
	- Different datasets
	- Different protein fitness measurements

	Configuration
	- Collaboration: 
		- Shared data stores, shared models, ability to define various model configurations.
	- Reproducibility:
		- Deterministic runs:
			- Shared and clear dataset definitions and configuration
			- Clear pipeline for each configuration option.

Issues for reference:
1. Hard-coded items: model architecture, dataset, and output - comparison is not currently possible.
2. Model and code is local only - Difficulty in collaboration.
3. (reproducibility may be an an issue later, note keeping track of rng data loading and training)

Solutions for reference:
1. Model configuration definitions.
2. Server-side data and model runs.
3. Git code repository.
```

```
Notes for reference:

Modify text: 
- The solution text is the most important. 
- Add specific detail (e.g. where in the code base does the configuration apply? What does model running on the server look like?)
- Some points can be made more clear: Why does server-side running enable organized collaboration?
- What does collaboration entail? Model running? Model architecture definitions? Likely the latter.
```