# -EFEO
Mask-R-CNN

ARCHEOLOGICAL MAPPING WITH DEEP LEARNING
EFEO
Summary:
I°] SOTA (state of the art) bibliographical study on the different existing technologies
2°] Why we whose Mask R-CNN
3°] How to apply it in our case
4°] future improvement and how to deploy it on a real case study
5°] Conclusion

1°]SOTA

Bibliography:
[1]: Verschoof-van der Vaart, W. B., & Lambers, K. (2019). Learning to look at LiDAR: The use of R-CNN in the automated detection of archaeological objects in LiDAR data from the Netherlands. Journal of Computer Applications in Archaeology, 2(1), 31–40. https://doi.org/10.5334/ jcaa.32

[2] : Bonhage A, Eltaher M, Raab T, Breuß M, Raab A, Schneider A. A modified Mask region-based convolutional neural network approach for the automated detection of archaeological sites on high-resolution light detection and ranging-derived digital elevation models in the North German Lowland. Archaeological Prospection. 2021; 1–10.
[3] : Trier, Ø. D., Reksten, J. H., & Løseth, K. (2021). Automated mapping of cultural heritage in Norway from airborne lidar data using faster R-CNN. International Journal of Applied Earth Observations and Geoinformation, 95. https://doi.org/10.1016/j.jag.2020.102241
[4] : Guyot, A, Lennon, M, Lorho, T and Hubert-Moy, L. 2021. Combined Detection and Segmentation of Archeological Structures from LiDAR Data Using a Deep Learning Approach. Journal of Computer Applications in Archaeology, 4(1), pp. 1–19. DOI: https://doi. org/10.5334/jcaa.64
[5] : Davis, D.S.; Lundin, J. Locating Charcoal Production Sites in Sweden Using LiDAR, Hydrological Algorithms, and Deep Learning. Remote Sens. 2021, 13, 3680. https:// doi.org/10.3390/rs13183680
[6] : 

-	R-CNN and Faster R-CNN (Trier, Verschoof)
-	Mask R-CNN (Guyot, Bonhage)
-	U-net (Bundzel, Banaziak, Trotter)
-	YOLOv3, YOLOv4 (Olivier)
Evolution: progress made in recent years is almost completely linked to the basic CNN-based model and its extensions, namely, the region-based CNN (R-CNN), the fully convoluted network, the Fast R-CNN and the more efficient variant Faster R-CNN. Mask R-CNN expands the Faster R-CNN architecture by adding an algorithmic branch for predicting an object segmentation mask parallel with the existing region proposal stage. U-Net which does semantic segmentation of image data.
A recent development in deep neural networks for object detection in natural images is the region-proposing convolutional neural network (R-CNN; Girshick et al., 2014), which may also be used for cultural heritage detection in ALS data. Verschoof-van der Vaart and Lambers (2019) use Faster R-CNN (Ren et al., 2017) to detect prehistoric barrows and Celtic fields in ALS data from the Netherlands. He et al. (2017) extend Faster R-CNN into Mask R-CNN by providing, for each detected object, an object mask in addition to the bounding box provided by Faster R-CNN.
R-CNN and Faster R-CNN used by Verschoof-van der Vaart and Lambers: Learning to Look at LiDAR, and by Ø.D. Trier et al.  Methode detailed but algorithm used less detailed on the internet and the exact algorithm used doesn’t exist anymore on github for Verschoof. For Trier it is : chenyuntc/simple-faster-rcnn-pytorch: A simplified implemention of Faster R-CNN that replicate performance from origin paper (github.com)
Yolov4 used by Olivier, exist on github and well described.
Mask R-CNN are used by Guyot and al, bonhage and al. well described and lot of explanations on how to train it on the internet, code on github. Mask R-CNN have the big advantage of having their first layers already pretrained (transfer learning) compared to yolo, so we need much fewers images to train them. And also are the closest to what we want to acheive visually. Because the mask takes the form of the object detected and the final result is an image with the object detected in color compared to the other methods which just place a square around the object detected. 
A reoccuring pattern in the articles using CNN for archeology is that algorithms are used to detect at most 2 types of object (only one type in bonhage and al) so the distinction between the different classes is not really a problem.
Bonhage focuses more on the pretreatment of the images to make them easier to treat by the algorithm than on fixing to algorithm to make it more efficient, because it only tries to detect one type of object. Bonhage uses licensed algorithms while Guyot don’t.
Good article by Abdullah
Splash of Color: Instance Segmentation with Mask R-CNN and TensorFlow | by Waleed Abdulla | Matterport Engineering Techblog

Article explaining the difference between R-CNN and YOLO :
R-CNN, Fast R-CNN, Faster R-CNN, YOLO — Object Detection Algorithms | by Rohith Gandhi | Towards Data Science

The limitation of YOLO algorithm is that it struggles with small objects within the image, for example it might have difficulties in detecting a flock of birds. This is due to the spatial constraints of the algorithm. But two-shot detectors such as the Faster R-CNN are significantly slower and more computationally intensive than one-shot detectors like YOLOv3, but they also can be more accurate. The two-shot detection model has two stages: region proposal and then classification of those regions and refinement of the location prediction. Single-shot detection skips the region proposal stage and yields final localization and content prediction at once. Article explaining the difference : The Battle of Speed vs. Accuracy: Single-Shot vs Two-Shot Detection Meta-Architecture - ClearML

Abdulla’s code to train our own nn :
GitHub - matterport/Mask_RCNN: Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow

U-net is used by Bundzel, more advanced of complicated to implemant, said to perform better than Mask-R-CNN. Pretrained on lunar LIDar data. Interesting point : « the area contains agricultural fields, roads and other important ancient infrastructure. Because every category has different features, we have decided to build a separate model for each one ». Bundzel work is very complete and technical, could be used later but in 3 months i really doubt that i could be able to come close to it.

In order to derive additional metrics, we split all the annotated objects into 3 categories: True-positive (TP)—existing objects, found by a model. False-positive (FP)—not existing objects, wrongly predicted by a model. False-negative (FN)—existing objects, not found by a model. The additional metrics we were interested in are Precision, Recall, and F-score. Precision is a fraction of true-positive objects among all the predicted objects, recall is a fraction of true-positive objects among all the existing object, and F-score is a harmonic mean of those two.


Avancement
-algorithm made by abdullah understood and tried but it needs a GPU to work which I don’t have
-found another algorithm that works online so access to free GPU
-train on one batch of 50 images with only one classification on 15 epochs
-recall and precision for those epochs
-training on a larger batch with different annotations to have a better recall
-keys to have a better fit, zoom in to have fewer details on the images, 



 
Résultats : Version0 (5 epochs entrainés sur un premier test d’annotations non exhaustive)
TOT=nombre d’objets présents sur les images au total
OF=nombre d’objets trouvés par l’algo (en comptant les faux ET les vrais)

#comptés à la main
TOT=1092

#calculés en faisant tourner la cellule précedente sur les 5 modèles
OF1=565
OF2=365
OF3=341
OF4=282
OF5=358

# comptés à la main
FP1=68
FP2=28 
FP3=7
FP4=6
FP5=16
precision de l'epoch 1 =  0.879646017699115
recall de l'epoch 1 =  0.4551282051282051
F_score de l'epoch 1 =  0.59987929993965
precision de l'epoch 2 =  0.9232876712328767
recall de l'epoch 2 =  0.3086080586080586
F_score de l'epoch 2 =  0.4625943719972546
precision de l'epoch 3 =  0.9794721407624634
recall de l'epoch 3 =  0.3058608058608059
F_score de l'epoch 3 =  0.4661549197487788
precision de l'epoch 4 =  0.9787234042553191
recall de l'epoch 4 =  0.25274725274725274
F_score de l'epoch 4 =  0.40174672489082974
precision de l'epoch 5 =  0.9553072625698324
recall de l'epoch 5 =  0.3131868131868132
F_score de l'epoch 5 =  0.4717241379310345
 

moyenne de la précision 0.9432872993039213
moyenne recall 0.3271062271062271
moyenne F_scores 0.48041989090150955


pour voir si la taille des anchors n’est pas une limite :
max objects trouvés sur epoch1 =  16 /content/dataset/15201h.png
max objects trouvés sur epoch2 =  14 /content/dataset/15201h.png
max objects trouvés sur epoch3 =  12 /content/dataset/15201h.png
max objects trouvés sur epoch4 =  12 /content/dataset/15201h.png
max objects trouvés sur epoch5 =  11 /content/dataset/15201h.png

Résultats version1 (5epochs entrainés sur une nouvelle version des annotations plus exhaustive pour essayer d’augmenter le recall)
#comptés à la main
TOT=1092

#calculés en faisant tourner la cellule précedente sur les 5 modèles
OF1=631
OF2=458
OF3=374
OF4=358
OF5=401
On remarque déjà que le nombre d’OF (number of Object Found) a augmenté ce qui est bon signe

# comptés à la main
FP1=32
FP2=10 
FP3=7
FP4=6
FP5=5
Le nombre de FP est lui aussi augmenté ce qui est bon pour la precision
max objects trouvés sur epoch1 =  23 /content/dataset/15201h.png
max objects trouvés sur epoch2 =  18 /content/dataset/15201h.png
max objects trouvés sur epoch3 =  18 /content/dataset/15201h.png
max objects trouvés sur epoch4 =  17 /content/dataset/15201h.png
max objects trouvés sur epoch5 =  19 /content/dataset/15201h.png
precision de l'epoch 1 =  0.9492868462757528
recall de l'epoch 1 =  0.5485347985347986
F_score de l'epoch 1 =  0.6952988972721997
precision de l'epoch 2 =  0.9781659388646288
recall de l'epoch 2 =  0.41025641025641024
F_score de l'epoch 2 =  0.5780645161290322
precision de l'epoch 3 =  0.9812834224598931
recall de l'epoch 3 =  0.3360805860805861
F_score de l'epoch 3 =  0.5006821282401093
precision de l'epoch 4 =  0.9832402234636871
recall de l'epoch 4 =  0.32234432234432236
F_score de l'epoch 4 =  0.48551724137931035
precision de l'epoch 5 =  0.9875311720698254
recall de l'epoch 5 =  0.3626373626373626
F_score de l'epoch 5 =  0.5304755525787006
 
moyenne de la précision 0.9759015206267574
moyenne recall 0.395970695970696
moyenne F_scores 0.5580076671198705


On remarque une augmentation de la précision et du recall lorsqu’on augmente le nombre d’annotations (il est donc nécessaire d’être le plus exhaustif possible sur les annotations)

J’aimerais montrer maintenant si l’augmentation du nombre d’epochs peut aider aussi à augmenter le recall

Bonhage: the model can identify RCH sites with an average recall of 83% and an average precision of 87%
But number of features to detect is 32

Une piste pour augmenter le recall est de diminuer le zoom des images pour avoir moins de features par images.

OF6=386
OF7=349
OF8=393
OF9=353
OF10=481
Le nombre d’OF augmente encore lorsqu’on augmente le nombre d’epoch (avec toujours un pic sur le dernier de la série ce que se produisait aussi lorsqu’on utilisait 5 epochs, je ne sais pas pourquoi)

precision de l'epoch 1 =  0.9492868462757528
recall de l'epoch 1 =  0.5485347985347986
F_score de l'epoch 1 =  0.6952988972721997
precision de l'epoch 2 =  0.9781659388646288
recall de l'epoch 2 =  0.41025641025641024
F_score de l'epoch 2 =  0.5780645161290322
precision de l'epoch 3 =  0.9812834224598931
recall de l'epoch 3 =  0.3360805860805861
F_score de l'epoch 3 =  0.5006821282401093
precision de l'epoch 4 =  0.9832402234636871
recall de l'epoch 4 =  0.32234432234432236
F_score de l'epoch 4 =  0.48551724137931035
precision de l'epoch 5 =  0.9875311720698254
recall de l'epoch 5 =  0.3626373626373626
F_score de l'epoch 5 =  0.5304755525787006
precision de l'epoch 6 =  0.9870466321243523
recall de l'epoch 6 =  0.3489010989010989
F_score de l'epoch 6 =  0.5155615696887685
precision de l'epoch 7 =  0.9885386819484241
recall de l'epoch 7 =  0.3159340659340659
F_score de l'epoch 7 =  0.47883414295628035
precision de l'epoch 8 =  0.9872773536895675
recall de l'epoch 8 =  0.3553113553113553
F_score de l'epoch 8 =  0.5225589225589226
precision de l'epoch 9 =  0.9830028328611898
recall de l'epoch 9 =  0.31776556776556775
F_score de l'epoch 9 =  0.48027681660899646
precision de l'epoch 10 =  0.9896049896049897
recall de l'epoch 10 =  0.4358974358974359
F_score de l'epoch 10 =  0.6052129688493325
moyenne de la précision 0.981497809336231
moyenne recall 0.3753663003663005
moyenne F_scores 0.5392482756261653
 

L’augmentation est pas flagrante mais on augmente quand même sensiblement le recall jusqu’à atteindre un recall de 0.43 à l’epoch 10.

Performances optimales à l’epoch 10 avec perfo 0.99 et recall 0.43

Prochains test est de diminuer la taille des images pour augmenter le recall !

 
