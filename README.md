# **Documentație Proiect: Model Bayes**


## **Componență Echipă**

- Cotigă David-Gabriel
- Dănăilă Mihai-Teodor
- Roșianu Radu-Daniel

## **Descriere succintă**

Proiectul nostru este un Clasificator Bayes care utilizează rezumatele filmelor (sinopsisurile) pentru a prezice genurile filmelor. Modelul analizează cuvintele din sinopsisuri, calculează probabilitățile pentru fiecare gen și folosește aceste informații pentru a clasifica filmele în funcție de genurile lor, cum ar fi dramă, aventură sau comedie.

## **Instalarea librăriilor necesare**

Pentru ca aplicația să funcționeze este nevoie de câteva librării externe. Astfel, rulăm următoarele comenzi:

```python
pip install pandas
pip install numpy
pip install nltk
pip install requests
pip install scikit-learn
```

## **Instrucțiuni de utilizare**

Pentru a rula programul rulăm următoarea comanda în folderul proiectului:

```python
python ./main.py
```

Programul afișează acuratețea modelului (conform setului de date destinate testing-ului), urmând ca user-ul sa aibă posibilitatea de a testa modelul folosind rezumatul oricărui film dorește. Pentru aceasta, trebuie introdus numele filmului dorit în engleză. Programul va afișa sinopsis-ul filmului găsit (prin baza de date OMDb) și genurile atribuite acestuia, urmate de genurile prezise de model.

## **Exemplu de utilizare cu filmul The Godfather**

```bash
Introduceti numele unui film: **The Godfather**

Sinopsis film: The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.

Genuri film: ['Crime', 'Drama']
```

Modelul a prezis:

- crime
- drama
- comedy

```bash
One more time?: (y/n)
n
```

În acest exemplu, programul a identificat corect genurile crime și drama, însă a sugerat și comedy, care nu este un gen asociat acestui film.

## **Explicarea Modelului Bayes pentru Clasificarea Genurilor Filmelor**

Modelul prezentat este un clasificator Bayes, un algoritm de clasificare probabilistic care este folosit aici pentru a prezice genul unui film pe baza sinopsisului său. Modelul Bayes este un model popular în procesarea limbajului natural (NLP) datorită simplității și eficienței sale, chiar și în cazul unor seturi de date mari. În continuare, vom detalia fiecare componentă a acestui model și vom explica modul în care acesta funcționează din punct de vedere matematic pentru a determina genul probabil al unui film.

Setul de date utilizat atat in testarea, cat si in antrenamentul modelului a fost obținut de pe platforma Kaggle (destinata dataset-urilor pentru machine learning), si contine date (inclusiv sinopsis-uri) despre primele 1000 cele mai bine cotate filme de pe IMDb.

### **1. Probabilitățile a priori (P(gen))**

Modelul începe prin calcularea probabilităților a priori pentru fiecare gen de film, adică șansele ca un film să aparțină unui anumit gen, independent de sinopsis. Aceste probabilități sunt calculate pe baza proporției filmelor din setul de date de antrenament care sunt asociate fiecărui gen. De exemplu, dacă 20% dintre filme sunt de acțiune, atunci $P(acțiune) = 0,2.$

Formula utilizată este:

$$
P(gen) = (nr\ filme\ din\ acel\ gen) / (nr\ total\ filme\ din\ setul\ de\ antrenament)
$$

### **2. Probabilitățile condiționate (P(cuvânt | gen))**

Pentru a evalua probabilitatea ca un film să aparțină unui anumit gen, modelul folosește probabilitățile condiționate P(cuvânt | gen). Acestea exprimă probabilitatea ca un cuvânt specific să apară într-un sinopsis, dat fiind că filmul este dintr-un anumit gen. Modelul folosește aceste probabilități condiționate pentru a lua în considerare frecvențele diferitelor cuvinte în genurile distincte.

Formula utilizată cu Laplace smoothing este:

$$
P(cuvânt | gen) = (nr\ apariții\ al\ cuvântului\ în\ sinopsisurile\ acelui\ gen\ + 1) / (nr\ total\ de\ cuvinte\ din\ acel\ gen\ + dimensiunea\ vocabularului)
$$

Adăugarea '+1' este necesară pentru a evita o probabilitate de zero în cazul cuvintelor care nu apar în anumite genuri din setul de antrenament. Dimensiunea vocabularului adaugă stabilitate la numitor, asigurând că suma probabilităților pentru toate cuvintele se păstrează.

Fără **Laplace smoothing**, dacă un cuvânt din rezumatul unui film nu apare în setul de antrenament pentru un gen, probabilitatea sa pentru acel gen va fi calculată drept zero. De exemplu, dacă un film cu cuvântul „spaceship” nu se încadrează în „dramă” (unde cuvântul nu apare), modelul va calcula o probabilitate zero pentru „dramă”. Acest lucru înseamnă că genul „dramă” va fi exclus automat, chiar dacă alte indicii din sinopsis sugerează că ar putea fi relevant. Cu Laplace smoothing, atribuim o probabilitate mică tuturor cuvintelor, chiar și celor noi, evitând astfel această problemă și permițând modelului să considere toate genurile posibile.

### **3. Scorul de probabilitate pentru fiecare gen (log P(gen | sinopsis))**

Pentru a prezice genul unui film, algoritmul calculează scorul de probabilitate pentru fiecare gen. Acest scor este produsul dintre probabilitatea a priori a genului P(gen) și probabilitățile condiționate pentru fiecare cuvânt din sinopsisul dat. În practică, se utilizează logaritmi pentru a transforma produsul probabilităților într-o sumă de logaritmi, ceea ce simplifică calculul și previne probleme de underflow (valori prea mici pentru a fi stocate în mod precis în memori~~e~~a calculatorului).

Formula pentru calcularea scorului este:

$$
Scor(gen) = log(P(gen)) + \sum(log(P(cuvânt | gen))) pentru fiecare cuvânt în sinopsis
$$

Astfel, fiecare gen primește un scor logaritmic, iar genurile cu cele mai mari scoruri sunt considerate cele mai probabile pentru sinopsisul dat. Logaritmul schimbă produsele în sume, ceea ce ajută la evitarea pierderilor numerice și face modelul mai eficient în procesarea datelor textuale complexe.

Fără a calcula probabilitățile în mod logaritmic întâmpinăm o problemă legată de scăderea preciziei numerice atunci când multiplicăm probabilitățile, mai ales când avem mai multe cuvinte (sau caracteristici) într-un set de date foarte mare. Acest lucru se datorează faptului că probabilitățile vor fi calculate drept numere foarte mici, iar înmulțirea lor poate duce la valori și mai mici, care sunt aproape imposibil de gestionat numeric, iar în unele cazuri pot duce la eroare de subnivel (underflow), adică pierderea de precizie. Pe scurt, folosirea de **logarithmic score** este o modalitate de floating point error mitigation.

### **4. Predicția**

După calcularea scorurilor logaritmice pentru fiecare gen, modelul selectează genurile cu cele mai mari scoruri ca predicții pentru sinopsisul respectiv. Aceasta se realizează prin sortarea scorurilor și alegerea genurilor de top. În acest fel, modelul poate prezice genurile probabile pentru un sinopsis dat, iar utilizatorul poate ajusta numărul de genuri pe care modelul le va returna, pe baza parametrului top_n (ex. top 3 genuri).

### **5. Evaluarea modelului**

Pentru a determina performanța modelului, se utilizează două metrici de acuratețe, care ajută la măsurarea cât de bine modelul poate prezice genurile filmelor noi. Aceste metrici sunt:

- **Acuratețea** – determină procentul de predicții corecte. Este considerată corectă o predicție dacă cel puțin unul dintre genurile prezise se potrivește cu unul dintre genurile reale ale filmului.
- **Exact-match accuracy** – determină procentul de predicții exacte. Aceasta înseamnă că modelul prezice exact toate genurile reale ale unui film, fără a omite sau adăuga genuri necorespunzătoare.

Aceste calcule ne permit să evaluăm cât de bine poate modelul să prezică genurile filmelor noi pe baza sinopsisurilor lor, și să îmbunătățim performanța sa pe baza rezultatelor obținute în aceste evaluări.

În general, la acuratețea normală rata de succes e în jur de 90%, iar la cea exactă în jur de 15%

## **Structura Codului**

Structura acestui cod este bine organizată în funcții specifice, fiecare având un rol clar în implementarea unui clasificator Bayes care prezice genul filmelor pe baza sinopsisurilor. Codul începe cu descărcarea resurselor din nltk, precum „stop words” și instrumentele de ,,lemmatizare’’, necesare pentru procesarea textului.

### **Funcții importante pentru calcul**

Setul de date cu filme este încărcat și preprocesat de funcția **`init_dataframe`**, care folosește un fișier CSV cu titluri de filme, genuri și sinopsisuri. Funcția filtrează coloanele relevante și aplică procesarea textului pe genuri și sinopsisuri, asigurând o formatare standardizată. După aceasta, setul de date este împărțit într-un set de antrenament și unul de testare, cu o proporție de 80-20%, pentru a permite evaluarea ulterioară a modelului.

Funcția **`preprocess_text`** aplică procesarea textului asupra fiecărui sinopsis sau gen, realizând conversia textului la litere mici, eliminarea semnelor de punctuație și a „stop words”, precum și aplicarea lemmatizării. Această procesare inițială este critică pentru a obține cuvinte relevante și unitare, asigurând că modelul nu va fi derutat de forme gramaticale sau cuvinte inutile. Rezultatul este o listă de cuvinte preprocesate care va fi folosită de clasificator pentru a învăța caracteristicile textului.

Probabilitățile a priori pentru fiecare gen sunt calculate de funcția **`calculate_apriori_prob`**, pe baza frecvenței fiecărui gen în setul de antrenament. Aceste probabilități reflectă distribuția genurilor și reprezintă probabilitatea de bază ca un film să aparțină unui gen anume, independent de sinopsis. Acestea vor fi combinate ulterior cu probabilitățile condiționate ale cuvintelor pentru a obține predicția finală a genurilor.

Funcția **`calculate_cond_prob`** calculează probabilitățile condiționate pentru fiecare cuvânt în funcție de genuri, folosind o metodă numită Laplace smoothing pentru a evita valorile de probabilitate zero pentru cuvintele noi. Această funcție transformă sinopsisurile într-un vector de cuvinte și calculează de câte ori apare fiecare cuvânt într-un anumit gen folosind funcția **CountVectorizer**. Concret, CountVectorizer convertește fiecare sinopsis preprocesat într-un vector numeric bazat pe frecvența fiecărui cuvânt din sinopsisurile filmelor. În final, construiește un dicționar cu probabilitățile condiționate ale cuvintelor pentru fiecare gen, pe baza frecvențelor acestora în setul de antrenament.

Funcția **`predict_top_genres`** folosește formula lui Bayes pentru a calcula scorurile de probabilitate logaritmică pentru fiecare gen. Aceasta pornește de la probabilitățile a priori și adaugă scorurile logaritmice pentru fiecare cuvânt din sinopsis, obținând astfel o valoare de scor pentru fiecare gen. Genurile cu cele mai mari scoruri sunt returnate ca fiind predicțiile modelului pentru genurile sinopsisului dat.

Pentru a evalua performanța modelului, funcția **`evaluate_on_test_set`** aplică clasificatorul pe setul de testare. Aceasta compară genurile prezise cu cele reale și returnează o listă cu predicțiile și genurile corecte pentru fiecare sinopsis. Această evaluare permite calcularea ulterioară a acurateței modelului și analizarea performanței acestuia în prezicerea corectă a genurilor.

Funcțiile de calcul al acurateței, **`calculate_exact_match_accuracy`** și **`calculate_accuracy`**, măsoară cât de bine a prezis modelul genurile filmelor. Prima funcție evaluează potrivirile exacte între predicții și genurile reale, în timp ce a doua funcție consideră o predicție corectă și atunci când modelul ghicește cel puțin unul dintre genurile corecte. Aceste măsurători oferă o imagine mai completă asupra preciziei modelului.

Funcția **`predict_movie_genres`** permite utilizatorului să introducă un titlu de film și să vadă genurile prezise de model. Folosește funcția get_movie_data pentru a obține sinopsisul și genurile reale de pe baza de date OMDb. Sinopsisul este prelucrat și clasificat, iar predicțiile genurilor sunt afișate utilizatorului alături de genurile reale, facilitând o evaluare rapidă.

În final, codul include un ciclu while care permite utilizatorului să introducă titluri de filme până când alege să oprească programul. În cadrul fiecărei iterații, se apelează predict_movie_genres pentru a afișa genurile prezise, ceea ce face ca programul să fie interactiv și să ofere feedback instantaneu.

## **Bibliografie**

[https://en.wikipedia.org/wiki/Scoring_rule](https://en.wikipedia.org/wiki/Scoring%5C%5C_rule)

[https://en.wikipedia.org/wiki/Naive_Bayes_classifier](https://en.wikipedia.org/wiki/Naive%5C%5C_Bayes%5C%5C_classifier)

[https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)

[https://en.wikipedia.org/wiki/Additive_smoothing](https://en.wikipedia.org/wiki/Additive%5C%5C_smoothing)
