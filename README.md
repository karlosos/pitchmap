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

## Notatki

- mógłbym na początku wyświetlać przyporządkowanych graczy do drużyn i użytkownik by ręcznie przeciągał tych co zostali źle sklasyfikowani

Color transform:
```python
hsv = np.asarray(list(map(lambda color: colorsys.rgb_to_hsv(color[0] / 255.0,
                                                            color[1] / 255.0, color[2] / 255.0),
                          extracted_player_colors)))
```

bbox:

```
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
```

Versions:
- 0.9
    - kalibracja ręczna: Na początku sekwencji wciśnij przycisk "kalibracja" i wybierz 
    punkty charakterystyczne. Następnie kliknij transformuj aby zobaczyć transformację.
    Następnie ponownie przycisk "kalibracja" aby wystartować wideo. 
    Następnie powtórz czynność dla końcowego punktu z tą różnicą, że po transformacji kliknij
    jeszcze raz transformuj aby włączyć interpolację liniową
    - interpolacja liniowa pomiędzy ręcznie wybranymi sekwencjami
    - nie można przerywać i resetować transformacji
    - zapisuje pozycje piłkarzy i homografię w plikach
    - zapisuje model detekcji drużyny w pliku
    
## Dane wyjściowe

### pitchmap

`baltyk_starogard_1.mp4_PlayersListComplex_CalibrationInteractorMiddlePoint`

Zapisane są:

`[self.players_list.players, self.__calibration_interactor.homographies]`

Gdzie players_list.players to pozycje graczy na klatce wideo a homographies
to macierze translacji. 

Trzeba będzie rzutować pozycje z wideo na pozycję modelu. 

Użyć `calibrator.transform_to_2d(players, current_frame_homography)`

### manual tracking

Zapisywane w pliku

`data/cache/{self.video_name}_manual_tracking.pik`

Dane:

`[self.__players_list, self.homographies, self.calibrator]`

Czyli mamy obiekt players_list z którego można wyłuskać pozycje graczy 
(już na modelu boiska). Reszta zmiennych nie będzie potrzebna. 
Będę mógł też porównać macierze translacji ale bym musiał mieć jakiś sposób
porównywania macierzy.

#### Proces

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
