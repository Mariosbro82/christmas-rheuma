Blutsauerstoff
Cardiofitness 
Atemfrequenz 
Sechs-minuten- gehtest 
Ruheenergie
• Körperliche Leistung
• Bewegen
• Trainieren
• Stehen
• Aktivitätsenergie
• Stehminuten
• Trainingsminuten
• Schritte
• Strecke (Gehen und Laufen)
• Stehstunden
• Treppensteigen
• Cardiofitness
• Trainings
• Cardioerholung
• Laufgeschwindigkeit
• Laufleistung
Bewegungsminuten 


Herzfrequenz
• Herzfrequenzvariabilität
• Ruheherzfrequenz
• Elektrokardiogramme (EKG)
• Cardiofitness
• Ø-Herzfrequenz (Gehen)
• Cardioerholung
• Mitteilungen für Cardiofitness
• Hohe Herzfrequenz-Mitteilungen



•Umgebungsgeräuschpegel
• Geräuschmitteilungen









•Bipedale Abstützungsdauer
• Gehtempo
• Schrittlänge im Gehen
• Treppensteigen: Abwärts
• Asymmetrischer Gang
• Cardiofitness
• Treppensteigen: Aufwärts



Schlaf:

Vitalzeichen
Schlaf score
Rem  phase 
Wach phase 
Kern phase
Tief phase 


Trainings app : 
* Bruned calouris
* Anstrenrung 



Seelisches wohlbefinden: 

Zeit am tages licht 
Traningszeit in min 











Wetter: 

•Temperatur 
•Luftfeuchtigkeit
•Luftdruck
•Luftqualität


App eigene In app logik die wir brauchen für einen Wetter change in put : 

      Temperatur     + Wetter art ( schnee, Regen, Sonne etc ) + Luftdruck + Luftfechtigkeit... diese intern über die letzten 7 Tagen speichern ( alle 6 Stunden ) 
Und dann daraus immer einen Wetter change score bauen, welcher dann  in  als ein weiter Daten Punkt exestiert.. 



•Stress  ( in app logik that validates stress levels, from other data points in combination ) and then gives it as another fata strean into ml 

* Physio / übungen --> wie man sich gefühlt hat und der verlauf davon ( backend system again ) before it goes into ML 

•• Medikamente, welche art von Medikamente--> auch in bezug auf wirksweise, wegen nebenwirkungen 

•• Medikamente einnahme 

•• Journal: Everage Mood and the other sldiers on the top 

* Body pain map : Joints, Schweregrad, Vermutete trigger, schmerzens Art, 
* Flare Aufzeichnung: Flare punkt, flare schwerere, Flare auslöser. flare art, flare länge 
* Quik log
* Quik mood log 
* BASDAI 
* Other specific assements( but via univerals score and time score, so the ml is feed with a score of 1-10 based on the worsesness, no matter which assesment) And a time weighted one which than als takes the current score in ratio to the past scores. 





Sleep Schedule Consistency: Nicht wie lange man schläft, sondern die Variabilität der Einschlaf- und Aufwachzeiten.

Jahreszeit faktor 

Hauptmetrik (Top Level)
• Disease Activity Composite
Mood & Mental Health (Stimmung & Psychische Gesundheit)
• Mood Score Current
• Mood Valence Avg
• Mood Stability Index
• Anxiety Level
• Stress Level
• Stress Resilience Score
• Mental Fatigue Score
• Cognitive Function Score
• Emotional Regulation Score
• Social Engagement Level
• Mental Wellbeing Composite
• Depression Risk Indicators
Disease-Specific Assessments (Krankheitsspezifische Bewertungen)
• Basdai Score
• Asdas Crp Score
• Basfi Functional Index
• Basmi Mobility Index
• Patient Global Assessment
• Physician Global Assessment
• Tender Joint Count
• Swollen Joint Count
• Enthesitis Score
• Dactylitisscore
• Spinal Mobility Score
Pain Characterization (Schmerzcharakterisierung)
• Pain Intensity Current
• Pain Intensity Avg 24H
• Pain Intensity Max 24H
• Nocturnal Pain Severity
• Morning Stiffness Duration
• Morning Stiffness Severity
• Pain Location Count
• Pain Quality Burning
• Pain Quality Aching
• Pain Quality Sharp
• Pain Interference Sleep
• Pain Interference Activity
• Pain Variability Coefficient
• Breakthrough Pain Episodes
