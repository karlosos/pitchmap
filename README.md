<p align="center">
  <img src="https://i.imgur.com/yyO9y7W.png" alt="pitch_map_logo" />
</p>

# PitchMap

Oprogramowanie do śledzenia ruchu piłkarzy na boisku na podstawie obserwacji przez ruchomą kamerę.

A software for the tracking of football players on the playfield captured by a moving camera.

## Cel

Celem pracy jest stworzenie algorytmu i aplikacji do wizualnego śledzenia ruchu poszczególnych piłkarzy na boisku i rzutowania trajektorii ich ruchu na płaszczyznę boiska. Przyjęto założenia, że obserwacja boiska odbywa się za pomocą ruchomej kamery (dopuszczalne są jedynie ruchy w zakresie zmian punktu obserwacji, inne parametry kamery są stałe). Podczas tworzenia oprogramowania wykorzystać można gotowe biblioteki z zakresu przetwarzania i rozpoznawania obrazów oraz widzenia komputerowego.

## Zakres pracy

Zakres pracy:

* Przegląd literatury z zakresu przetwarzania i rozpoznawania obrazów oraz wykrywania i śledzenia obiektów w strumieniu video;
* Wybór odpowiednich algorytmów;
* Przegląd funkcjonalności bibliotek zorientowanych na śledzenie obiektów;
* Opracowanie projektu i realizacja oprogramowania komputerowego;
* Testowanie opracowanego oprogramowania;
* Wnioski końcowe.

## Używanie

### Konfiguracja środowiska

Stwórz wirtualne środowisko za pomocą `virtualenv`. Jeżeli nie masz, zainstaluj przy pomocy `pip install virtualenv`.

```
virtualenv .venv
```

Po stworzeniu wirtualnego środowiska włącz je w terminalu:

```
source .venv/bin/activate
```

Na Windowsie komenda ta może wyglądać inaczej. Na Windowsie trzeba włączyć skrypt z `.venv/Scripts/activate.bat`.

Zainstaluj wymagane zależności, które są opisane w `requirements.txt` za pomocą komendy:

```
pip install -r requirements.clean.txt
```

Pamiętaj aby wykonać to na aktywowanym wirtualnym środowisku, inaczej zainstaluje ci te wszystkie zależności globalnie.

Na windowsie do instalacji paczki `mxnet` wymagane są "Microsoft C++ Build Tools" (https://visualstudio.microsoft.com/visual-cpp-build-tools/).

Należy ustawić zmienną środowiskową:

```
$env:SM_FRAMEWORK="tf.keras"
```

lub za pomocą pycharma.

### Używanie aplikacji

W pliku `pitchmap/main.py` zmień plik wideo wejściowy. To znaczy zmień polę `self.video_name` w klasie `PitchMap`.
Materiały wideo powinny być umieszczone w folderze `data/`.

Włącz aplikację z poziomu folderu repozytorium. To znaczy wywołaj komendę:

```
python pitchmap/main.py
```

Pamiętaj aby włączać wszystkie skrypty z poziomu głównego folderu repozytorium. Inaczej importy i ścieżki nie będą się zgadzały.

Po włączeniu aplikacji przeprowadzi się analiza ruchu kamery za pomocą dense optical flow. Widoczne będzie małe okno.
Następnie przeprowadzona zostanie nauka klasyfikatora drużyny zawodników. Poczekaj cierpliwie. Po tym etapie nauczony klasyfikator
i dane z analizy ruchu kamery zostaną zapisane i zapiklowane. Zapiklowane dane będą dostępne w folderze `data/cache`.

Jak pojawi się interfejs użytkownika będziesz mógł przeprowadzić kalibrację. Wciśnij przycisk **Calibrate**. Oznacz punkt 
na klatce wejściowego wideo a następnie odpowiadający mu punkt na modelu boiska. Oznacz takich par minimum 4.
Po kliknięciu przycisku **Transform** zwizualizowane będzie przekształcenie geometryczne. Zatwierdź kalibrację przyciskiem
**Apply**. Powtórz te czynności dla każdej klatki. Ilość klatek zależna jest od wybranego sposobu kalibracji. Sposób kalibracji
można zmienić w linii `calib_interactor = calibrator_interactor.CalibrationInteractorMiddlePoint` w pliku `pitchmap/main.py`.
Możliwe kalibracje do wyboru to:

* CalibrationInteractorMiddlePoint - trzy klatki charakterystyczne,
* CalibrationInteractorAutomatic - dwie klatki charakterystyczne,
* CalibrationInteractorSimple - ręczny wybór klatek charakterystycznych.

Po przeprwoadzeniu kalibracji możesz włączyć/wyłączyć projekcję za pomocą przycisku **Toggle projection**. Aby zacząć 
detekcję przewiń wciśnij przycisk **Toggle detection** i przewiń materiał za pomocą suwaka na początek. Po skończeniu
detekcji zamknij program. Zebrane dane zostaną zapisane do folderu: `{nazwa filmu}_PlayersListComplex_{nazwa wybranej kalibracji}`.
*PlayersListComplex* jest to rodzaj przeprowadzanej identyfikacji gracza. Można przełączyć się na sposób simple za pomocą `player_list = player.PlayersListSimple`
w pliku `pitchmap/main.py`.

### Dane wyjściowe

Przykładowo w pliku:

`baltyk_starogard_1.mp4_PlayersListComplex_CalibrationInteractorMiddlePoint`

Zapisane są:

`[self.players_list.players, self.__calibration_interactor.homographies]`

Gdzie `players_list.players` to pozycje graczy na klatce wideo a homographies
to `macierze transformacji`. 

Aby używać tych danych będzie trzeba rzutować pozycje z wideo na pozycję modelu 
używając `calibrator.transform_to_2d(players, current_frame_homography)`.

## Ręczne zbieranie danych

Stworzono program `manual_tracking` który służy do ręcznego zbierania danych w celach
porównawczych. Te dane służą jako dane wzorcowe.

Należy skonfigurować program w `manual_tracking/main.py`, czyli określić materiał wideo. 
Po włączeniu programu należy przeprowadzić kalibrację za pomocą przycisku **Calibrate**. 
Po oznaczeniu punktów wcisnij przycisk **Transform** aby zatwierdzić kalibrację. Możesz 
przesuwać punkty trzymając prawy przycisk myszy. Aby przewinąć klatki używaj 
strzałek na klawiaturze. Po przejściu na następną klatkę punkty będą
w tym samym miejscu. Ma to przyśpieszyć operację kalibracji. Wystarczy przesuwać punkty
wcześniej oznaczone do nowych miejsc. 

Po przeprowadzeniu kalibracji dla każdej klatki należy oznaczyć piłkarzy.  Sugerowane jest
następujące postępowanie. W klatce pierwszej oznacz wszystkich zawodników. Przy oznaczaniu
od razu przyporządkuj kolory drużyn za pomocą przycisków (0, 1, 2) które są w odpowiednich
kolorach. Wyłącz aplikację,
w kodzie odkomentuj linie:

```
mt.players_list.fixed_player_id = 1
FAST_ADDING = True
```

Spowoduje to, że będziesz oznaczać tylko zawodnika o id 1 i klatki same będą się przełączać.
Przejdź na drugą klatkę. Zobaczysz przyciemnione pozycje piłkarzy na modelu boiska. Oznacz
pozycję piłkarza na klatce wejściowej, który w poprzedniej klatce miał id=1 i rób tak aż nie
wyjdzie on poza obszar widoku kamery. Dla kolejnego gracza zmień `fixed_player_id` na 2. 

Przy każdym wyjściu z aplikacji dane są zapisywane w pliku:

`data/cache/{self.video_name}_manual_tracking.pik`

O strukturze:

`[self.__players_list, self.homographies, self.calibrator]`

Czyli mamy obiekt `players_list` z którego można wyłuskać pozycje graczy 
(już na modelu boiska). Reszta zmiennych nie będzie potrzebna. Po włączeniu aplikacji
dane są ładowane. 

### Proces szybkiego oznaczania piłkarzy w manual tracking app

W pierwszej klatce oznaczyć wszystkich piłkarzy. Wyłączyć program.
Dodać wczytywanie kolejnej klatki po dodaniu gracza 

```
    FAST_ADDING = True
```

Ustawić id gracza którego chcemy dodawać:

```
    mt.players_list.fixed_player_id = 29
```

Aby dodać pojawiających się graczy jeszcze nie wykrytych:

1. Przewijam wideo aż znajdę gracza bez białej kropki na pitchview
2. Zwiększam id gracza -> ustawiam stały id
3. Zaznaczam piłkarza w każdej klatce

## Oznaczanie relacji pomiędzy danymi zebranymi manualnie i automatycznie

Gdy już zbierze się dane przy pomocy `pitch_map` i `manual_tracking` i chce się przeprowadzić
analizę ścieżek i porównanie należy oznaczyć relacje pomiędzy tymi zbiorami danych. W tym celu
włącz aplikację `python players_path_selector/main.py`. Wcześniej jednak zmień w kodzie
ścieżki do materiału wejściowego i porównywanych danych wejściowych. 

Na lewej połówce wyświetlane są dane zebrane w sposób manualny. Na prawej części dane zebrane
automatycznie. Oznacz na lewej klatce piłkarza, klikając lewym przyciskime. Oznacz na prawej
klatce odpowiadającego piłkarza. Program służy do oznaczania jednej pary, to znaczy tylko 
jednego zawodnika. Dane zapisywane są do pliku, który ma w swojej nazwie id zadeklarowane w
kodzie `players_path_selector.main.py` w linii:

``` 
mt.id = "testing"  # file name
```

Zebrane dane będą zapisane w pliku `baltyk_starogard_1.mp4_path_selector_middle.pik`.
Po włączeniu aplikacje dane się nadpisują, dlatego zmieniać `id` w pliku źródłowym.

## Porównanie danych

### Porównanie heatmap

Porównanie heatmap przeprowadzane jest za pomocą skryptu ```comparison/main.py```. 
Należy w klasie `Comparator` określić dane wejściowe, oraz w sekcji `__main__` nazwę
pliku wyjściowego. Plik wyjściowy ma zapisywać wyliczone heatmapy. Wynikiem porównań 
będą obrazy (heatmap) oraz wynik SSIM porównania do heatmapy z danych zebranych manualnie.

### Porównanie ścieżek

Porównanie ścieżek przeprowadza się za pomocą skryptu ```comparison/path.py```. Generuje ona wykresy ścieżek 
oraz mapy ścieżek zawodnika. Aby wygenerować
metrykę w postaci zmodyfikowanej odległości hausdorffa używa się pliku ```compatison/players_path_metric.py```.