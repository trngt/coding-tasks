# Draft Response:

-----------

Draft text:

### The project, its strengths, and the new requirements

This transformer-based prototype--that predicts protein fitness from sequence--is a strong foundational project, with potential for scaling up to a comprehensive research project. The code has a clear workflow with well-encapsulated components. These components create a nice scaffold for defining the model architecture, loading the data, and training and testing the model. Additionally, the creator has reproducibility in mind. They have set a random seed for consistent data creation and created a requirements.txt for consistent python package environments. For these reasons, the model is a nice starting point to scale up to more complex applications. Scaling will require three high-level ideas: model run variation for comparison (e.g., architecture, datasets, etc.), project setup for collaboration, and reproducible model runs for publication. However, these requirements also mean addressing a few notable issues with the project.

### Projects issues

The issues with the project include: (1) hard-coded configuration, (2) local storage of data, and (3) an incomplete environment definition for reproducibility. First, the model architecture, data source, and output are all hard-coded. So, we can't yet run different model variations nor compare run outputs. Second, the data storage and run output are both stored locally. This will create organizational issues when we begin collaboration---as data sets can misalign, and model runs will vary across disparate machines. Finally, while the requirements.txt is a good starting point, the python version and other machine-specific packages are not specified and will likely vary across user machines. We will need a solution that addresses these issues before we can extend this project.

### Proposal solutions

The solution, I propose, will include three main ideas: (1) model run configurations, (2) server-side data storage, and (3) code version control. First, we will create a way to define model run configurations, including architecture, attention mechanisms, and data storage. Using this configuration, we can create a simple way to run the model with various configurations. The output naming and storage will also reflect these varying configurations for run comparisons. Second, store the data and run outputs server-side to allow for collaboration. Researchers can define and implement different run configurations and run the model server-side, storing the output in a unified place for comparison. This idea is also linked to reproducibility, as a unified location for storing inputs and outputs will create a clear, reproducible pipeline from data loading to final model run output. Finally, store the project code on a server-side code repository. This will allow collaborators to define and implement configurations to the model server-side. Importantly, it is critical to include a git identifier (commit hash/version tag) on model run outputs for reproducibility. This solution unifies shared components and creates a configurable workflow for a streamlined collaborative research process.

```

Issues for reference:
1. Hard-coded items: model architecture, dataset, and output - comparison is not currently possible.
2. Model and code is local only - Difficulty in collaboration.
3. Reproducibility: environment specification (conda or Docker)

Solutions for reference:
1. Model configuration definitions.
	(model definitions, yml or json configuration files)
2. Server-side data and run environment.
	- Necessity of server-side data (expand)
	- Run environment enforces strict reproducibility, and aids in collaboration.
3. Git code repository.
	- Collaboration across model definitions and configuration.
	- Clear workflow and branching
	- Clear tagging/hash commits for model runs for reproducibility
```

```
Notes for reference:

High-level ideas to incorporate:
1. Server-side data reasoning. (Add in solution)
2. Server-side running (Add in solution)
3. Need for Docker or conda environment for consistent server-side running.

Expand the solutions to a fourth paragraph, decide on which details to split.

Outstanding questions: 
- Where in the code base does the configuration apply? 
- What does model running on the server look like?
- Why does server-side running enable organized collaboration?
- What does collaboration entail? Model running? Model architecture definitions? Likely both.
```
