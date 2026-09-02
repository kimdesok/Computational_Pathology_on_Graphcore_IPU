# Graphcore IPU Projects
## Tasks: camelyon17-wilds image classification 
## Resnet18 model & datasets:
>* Resnet18 model was tried first, being the smallest kind, to test a train model, a validation model and their weight transfer between the corresponding wrappers
>* A set of small datasets were randomly selected from the original camelyon17-wilds to speed up the training and avoid the OOM.
>* Some design features:<br>
                  `Host model`<br>
                       ↓<br>
              **Training executable** -> copyWeightsToHost()<br>
                                               ↓<br>
                                          `Validation model`   ->     copyWeightsToDevice()<br>
                                                                             ↓<br>
                                                                   **Inference executable**<br>

      


>* Mistakes? could be reviewed by an expert through the dashboard
<img width="907" height="718" alt="image" src="https://github.com/user-attachments/assets/51e1a40b-01be-4717-acab-8f8d62d312ab" />

(Coming soon...)
## Tasks: camelyon image classification
## ViT & ViT-MIL models & datasets:

## Tasks: Panda biopsy image classification
## ViT & ViT-MIL models & datasets:
