# deepracer-analysis

This is a set of notebooks and utilities to enable analysis of logs for AWS DeepRacer.

This project is a redo of analysis solutions provided in the
[AWS DeepRacer Workshops repository](https://github.com/aws-samples/aws-deepracer-workshops).

There are a few motivations leading a decision to reorganise the repository
* Having a community version of the project in a folder on a branch of a fork
of the original git repository has been causing issues when looking for it
* Jupyter notebooks are difficult to manage through source control as a generated file in
json format is hardly readable at all. This needed a new approach
* The project relied on a bunch of Python files which acted as a bag for code. There was
a need to extract it into a separate project to maintain the versioning and apply some
hopefully good practices

Separate repository makes it easier to find it. Jupyter combined with
[Jupytext](https://github.com/mwouts/jupytext) enables maintaining the notebook as
a set of Python files from which a notebook can then be genereted.
Finally, the project files have been moved to
[DeepRacer utils](https://github.com/aws-deepracer-community/deepracer-utils).

## Using the notebooks

The notebooks require Jupyter to run, together with deepracer-utils. While not needed
for using the notebooks, it's worth to also have Jupytext installed.

If you only plan to use the notebooks, I recommend that you make a copy of them to enable
seamless pulls of any updates.

If you pull latest changes for the notebooks, do also run
```
pip install --upgrade -r requirements.txt
```
in your venv. This way you will also get upgrades on the requirements.

## Running
```
python3 -m venv venv
pip install --upgrade -r requirements.txt
jupyter lab
```

## Modifying the notebooks
If you want to use the notebooks as a user and don't intend to submit changes,
simply use them through Jupyter Notebook or Jupyter Lab.

If you would like to submit changes to a notebook however, follow instruction in the
[Jupytext README](https://github.com/mwouts/jupytext) to enable pairing of the notebook
with a light script. This means that any changes you apply to the notebook or the .py
file paired with it will be synched.

When applying changes to the notebook, make sure you can use them with the sample log
resources and at the end of work restart the Kernel and run all the cells to provide
a clean view in the notebook.

## Roadmap
[x] Recreate the training and evaluation notebooks on top of the deepracer-utils
[ ] Apply changes from the original notebook commited since the creation of the log analysis branch
[ ] Redo log analysis challenge PRs and apply them to notebooks
[ ] Prepare simpler, specialised notebooks for everyday use
[ ] Prepare a tutorial on how to use and contribute to a notebook

## Credits
We would like to thank:
* AWS employees who have given birth to these tools as part of the
[AWS DeepRacer Workshops repository](https://github.com/aws-samples/aws-deepracer-workshops).
* All involved AWS DeepRacer Community members to contributed to its development

## License
This project retains the license of the 
[aws-deepracer-workshops](https://github.com/aws-samples/aws-deepracer-workshops)
project which has been forked for the initial Community contributions.
Our understanding is that it is a license more permissive than the MIT license
and allows for removing of the copyright headers.

Unless clearly sated otherwise, this license applies to all files in this repository.

## Troubleshooting

If you face problems, do reach out to the [AWS DeepRacer Community](http://join.deepracing.io).
Channel #dr-training-log-analysis has been created for this purpose.
When you face an issue, it is worth running `pip freeze` and saving the output as it may be
due to a specific version of the dependencies installed.

## Contact
You can contact Tomasz Ptak through the Community Slack: http://join.deepracing.io.

