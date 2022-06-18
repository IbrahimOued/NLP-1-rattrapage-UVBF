# NLP 1 rattrapage
Membre du groupe:
* OUEDRAOGO Ibrahim
* ILBOUDO Aziz

Lien vers le projet:
[https://github.com/IbrahimOued/NLP-1-rattrapage-UVBF](https://github.com/IbrahimOued/NLP-1-rattrapage-UVBF)
## Partie 1 Collecte des donnees

Le contenu des chaque document se fait en utilisant la librairie `PyPDF2`. Nous recherchons chaque fichier PDF pour en extraire le contenu et une fois ce contenu extrait, nous créeons un dataframe composé du nom du fichier et du contenu du fichier, plus précisement la dernière page.

## Partie 2 Processer les données

Dans le domaine du traitement du langage naturel (NLP), le prétraitement de texte est la pratique consistant à nettoyer et à préparer des données textuelles. J'utiliserai une bibliothèque de logiciels open source spaCy pour préparer les données pour l'analyse, mais d'autres bibliothèques telles que NLTK peuvent également être utilisées.

1. **Modèle pré-entrainé**

Nous avons télécharger 'fr_core_news_md' qui est un modèle pré-formé de spaCy. Le modèle peut être considéré comme un pipeline. Lorsque nous l'appelons sur un texte ou un mot, le texte passe par un pipeline de traitement. Cela signifie que si le texte n'est pas tokenisé, il sera alors tokenisé, et par la suite, différents composants (tagger, parser, ner etc.) seront activés. Tokéniser du texte signifie transformer une chaîne ou un document en plus petits morceaux (tokens).

Le composant le plus intéressant du pipeline est le tagger qui attribue des balises Part-Of-Speech (POS) basées sur le modèle de langue française de SpaCy pour obtenir une variété d'annotations. Une balise POS (ou balise de partie du discours) est une étiquette spéciale attribuée à chaque jeton dans un corpus de texte pour indiquer le type de jeton (est-ce un adjectif ? Ponctuation ? Un verbe ? etc.) et souvent aussi d'autres catégories grammaticales tels que le temps, le nombre (pluriel/singulier), les symboles, etc. Les balises POS sont utilisées dans les recherches de corpus et les outils et algorithmes d'analyse dans le texte.

2. **Suppression des tags non nécéssaire**

Nous pouvons utiliser les balises Part Of Speech pour prétraiter les données en supprimant les tags indésirables. En supposant que nous voulions supprimer tous les chiffres de notre texte, nous pouvons alors pointer sur une balise spécifique et la supprimer.

Notre travail sera fait sur la colonne ***Conent***, où on vas tokeniser, lemmatiser et supprimer les mots vides :

### Partie 3 Créer un dictionnaire et un corpus
 
Les deux principales entrées du modèle de sujet LDA sont le dictionnaire et le corpus :

**Dictionnaire**: l'idée du dictionnaire est de donner à chaque jeton un identifiant unique.
**Corpus**: Après avoir attribué un identifiant unique à chaque jeton, le corpus contient simplement chaque identifiant et sa fréquence (si vous voulez vous y plonger, recherchez alors Bag of Word (BoW) qui vous initiera à l'incorporation de mots).
On va appliquer l'objet *Dictionnary* de *Gensim* qui va attribuer à chaque token un identifiant unique et ensuite nous allons filtrer les tokens les moins fréquent et les plus fréquents, et également limiter le vocabulaire à un maximum de 1000 mots :


A ce stage, nous sommes maintenant prêts à construire le corpus en utilisant le dictionnaire ci-dessus et la fonction `doc2bow`. La fonction `doc2bow()` compte simplement le nombre d'occurrences de chaque mot distinct, convertit le mot en son identifiant de mot entier et renvoie le résultat sous la forme d'un vecteur creux :


### Partie 4 Création du modèle LDA

L'étape suivante consiste à entraîner le modèle d'apprentissage automatique non supervisé sur les données. J'ai choisi de travailler avec le LdaMulticore, qui utilise tous les cœurs du processeur pour paralléliser et accélérer la formation des modèles. Si cela ne fonctionne pas pour vous pour une raison quelconque, essayez la classe `gensim.models.ldamodel.LdaModel` qui est une implémentation équivalente, mais plus simple et à un seul cœur.

Lors de l'insertion de notre corpus dans l'algorithme de modélisation de sujet, le corpus est analysé afin de trouver la distribution des mots dans chaque sujet et la distribution des sujets dans chaque document.

En entrée, nous donnons au modèle notre corpus et notre dictionnaire d'avant ; de plus, nous avons choisi d'itérer 50 fois sur le corpus pour optimiser les paramètres du modèle (c'est la valeur par défaut). Je sélectionne le nombre de sujets à dix et les travailleurs à 4 (trouvez le nombre de cœurs sur votre PC en appuyant sur les touches ctr+shift+esc). La réussite est de 10, ce qui signifie que le modèle traversera le corpus dix fois pendant l'entraînement.

1. Quel est le nombre de sujets

Après avoir formé le modèle, la prochaine étape naturelle consiste à l'évaluer. Après avoir construit les sujets, un score de cohérence peut être calculé. Le score mesure le degré de similarité sémantique entre les mots les mieux notés dans chaque sujet. De cette façon, un score de cohérence peut être calculé pour chaque itération en insérant un nombre variable de sujets.

Une gamme d'algorithmes a été introduite pour calculer le score de cohérence ($C_v, C_p, C_{uci}, C_{umass}, C_{npmi}, C_a, \dots$). Travailler avec la bibliothèque gensim rend le calcul de ces mesures de cohérence pour les modèles thématiques assez simple. Personnellement, j'ai choisi d'implémenter $C_v$ et $C_{umass}$. Le score de cohérence pour $C_v$ va de 0 (incohérence complète) à 1 (cohérence complète). Les valeurs supérieures à $0,5$ sont assez bonnes, selon John McLevey (source : Doing Computational Social Science : A Practical Introduction By John McLevey). D'autre part, $C_{umass}$ renvoie des valeurs négatives.

Pour les graphes  nous parcourons simplement un nombre différent de sujets et enregistrons le score de cohérence dans une liste. Ensuite, nous traçons en utilisant seaborn.

Lorsque nous regardons les scores de cohérence en utilisant les algorithms $C_{umass}$ ou $C_v$ le meilleur est généralement le maximum. En regardant les graphes nous choisissons $10$ sujets.

### Partie 5 Visualisation des topics

Nous allons maintenant visualier les différents sujets et les mots associés. Le plotting dans le code represnte les 10 sujet les en cercle. Ils ont été obtenus en utilisant la méthode PCA de réduction de dimensionnalité. Le but est d'avoir une distance afin d'éviter les chevauchelents et de rendre chaque cercle unique. En survloant un cercle, différents mots sont affichés à droite en bleu et la fréquence estimée des termes dans le sujet sélectionné en rouge. LEs sujets les plus proches les uns des autres sont plus liés.

### Partie 7 Construction du modèle avec TF-IDF

Nous avons rajouté une étape intermédiaire pour extraire les termes les plus important en se basant sur TF-IDF. Nous avons défini un seuil de $0.005$ pour ignorer les termes qui apparaissent moins de $5%$ that appear in less than 1% of the documents