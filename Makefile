hf-login: 
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub: 
	huggingface-cli upload Fareskh12/Classifiers ./Model /Models --repo-type=space --commit-message="Sync App files"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
