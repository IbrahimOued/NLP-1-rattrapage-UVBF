# NLP 1 rattrapage

## Partie 1 Collecte des donnees

Le contenu des chaque document se fait en utilisant la librairie `PyPDF2`. Nous recherchons chaque fichier PDF pour en extraire le contenu et une fois ce contenu extrait, nous créeons un dataframe composé du nom du fichier et du contenu du fichier, plus précisement la dernière page.

## Partie 2 Processer les données

Dans le domaine du traitement du langage naturel (NLP), le prétraitement de texte est la pratique consistant à nettoyer et à préparer des données textuelles. J'utiliserai une bibliothèque de logiciels open source spaCy pour préparer les données pour l'analyse, mais d'autres bibliothèques telles que NLTK peuvent également être utilisées.

1. **Modèle pré-entrainé**

Nous avons télécharger 'fr_core_news_md' qui est un modèle pré-formé de spaCy. Le modèle peut être considéré comme un pipeline. Lorsque nous l'appelons sur un texte ou un mot, le texte passe par un pipeline de traitement. Cela signifie que si le texte n'est pas tokenisé, il sera alors tokenisé, et par la suite, différents composants (tagger, parser, ner etc.) seront activés. Tokéniser du texte signifie transformer une chaîne ou un document en plus petits morceaux (tokens).

Le composant le plus intéressant du pipeline est le tagger qui attribue des balises Part-Of-Speech (POS) basées sur le modèle de langue française de SpaCy pour obtenir une variété d'annotations. Une balise POS (ou balise de partie du discours) est une étiquette spéciale attribuée à chaque jeton dans un corpus de texte pour indiquer le type de jeton (est-ce un adjectif ? Ponctuation ? Un verbe ? etc.) et souvent aussi d'autres catégories grammaticales tels que le temps, le nombre (pluriel/singulier), les symboles, etc. Les balises POS sont utilisées dans les recherches de corpus et les outils et algorithmes d'analyse dans le texte.

1. **Suppression des tags non nécéssaire**

Nous pouvons utiliser les balises Part Of Speech pour prétraiter les données en supprimant les tags indésirables. En supposant que nous voulions supprimer tous les chiffres de notre texte, nous pouvons alors pointer sur une balise spécifique et la supprimer.

Notre travail sera fait sur la colonne ***Conent***, où on vas tokeniser, lemmatiser et supprimer les mots vides :

### Partie 3 Créer un dictionnaire et un corpus
 
Les deux principales entrées du modèle de sujet LDA sont le dictionnaire et le corpus :

**Dictionnaire**: l'idée du dictionnaire est de donner à chaque jeton un identifiant unique.
**Corpus**: Après avoir attribué un identifiant unique à chaque jeton, le corpus contient simplement chaque identifiant et sa fréquence (si vous voulez vous y plonger, recherchez alors Bag of Word (BoW) qui vous initiera à l'incorporation de mots).
On va appliquer l'objet *Dictionnary* de *Gensim* qui va attribuer à chaque token un identifiant unique et ensuite nous allons filtrer les tokens les moins fréquent et les plus fréquents, et également limiter le vocabulaire à un maximum de 1000 mots :


A ce stage, nous sommes maintenant prêts à construire le corpus en utilisant le dictionnaire ci-dessus et la fonction `doc2bow`. La fonction `doc2bow()` compte simplement le nombre d'occurrences de chaque mot distinct, convertit le mot en son identifiant de mot entier et renvoie le résultat sous la forme d'un vecteur creux :


### Partie 4 Création du modèle LDA

