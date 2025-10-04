Columns available in pre-hospital stage: Puls tastbar	Lebensbedrohliche Blutung	Befund	Hautbefund	psychischer Erstbefund	Alter 	Geschlecht	Bergungen	Alle ICD-Codes	Einteilung (reduziert)	Kategorie (reduziert)	Weitere Diagnosen	Herzrhytmusstörungen	Weitere Massnahmen	Wochentag	Monat	Tag oder Nacht	Ereignisort	Bezirk	PLZ	Gemeinde	Ort	Kanton	Land	Atemwegbefund	Zeitpunkt, Messart, Wert	Gesicht	Arme Lokalisation	Arme	Sprache	Gesicht Lokalisation	Strassennummer	Ereignisort2	Ort3	Institution	PLZ4	Strasse	erfolgreich Ossär	erfolgreich Venös	Zugänge	Bewusstseinlage	Auffälligkeiten	Beatmung	GCS	IBD Systolisch 	CO2	NIBD Diastolisch	Temperatur	VAS_on_scene	etCO2	NIBD Systolisch	HR	HF	IBD Diastolisch	AF	SPO2 Einsatzart
Blutstillung	Externes Pacing	NACA (nummerisch)	EKG 3-Kanal	Ist venös bestehend	Cincinnati FAST	Wundversorgung	Mit automatischer Reanimations-Hilfe	NACA	Defibrillation	EKG Telemetrie	Kardioversion	Ist Reanimation durchgeführt	EKG 12-Kanal	Ist ossär bestehend	Thoraxdrainage	Mitglieder mit Einsatzfunktion	Aktuelles Ereignis Hauptdiagnose	ICD-Code der Hauptdiagnose	Detail	Einteilung	Körperregion	Kategorie	Körperfläche	Lagerungen	Befund Atmung	Atemgeräusche	Kardioversionen	Externe Pacings	Defibrillationen	Externe Pacings detailliert	Sauerstoffabgaben	(Be)-Atmung Lichtreaktion (li)	Pupillenweite (re)	Arm (Empfindlichkeit re)	Bein (Empfindlichkeit li)	Arm (Bewegung re)	Bein (Bewegung li)	Pupillenweite (li)	Lichtreaktion (re)	Bein (Bewegung re)	Bein (Empfindlichkeit re)	Arm (Bewegung li)	Arm (Empfindlichkeit li)


Medications used are in columns: Alle Medikamente	Alle Medikamente detailliert 

Target: VAS_on_arrival < 3

Target population: primary missions of adult trauma patients that start with a VAS > 3 
data_df = data_df[data_df["VAS_on_scene"] > 3]

if restrict_to_trauma:
    n_non_trauma = data_df[data_df['Einteilung (reduziert)'] != 'Unfall'].shape[0]
    print(f'Excluded {n_non_trauma} non-trauma patients')

    # adult non-trauma patients
    n_adult_non_trauma = data_df[(data_df['Einteilung (reduziert)'] != 'Unfall') & (data_df["Alter "] >= 16)].shape[0]
    print(f'Excluded {n_adult_non_trauma} adult non-trauma patients')
    # pediatric non-trauma patients
    n_pediatric_non_trauma = data_df[(data_df['Einteilung (reduziert)'] != 'Unfall') & (data_df["Alter "] < 16)].shape[0]
    print(f'Excluded {n_pediatric_non_trauma} pediatric non-trauma patients')

    data_df = data_df[data_df['Einteilung (reduziert)'] == 'Unfall']

if restrict_to_primary:
    n_secondary = data_df[data_df['Einsatzart'] != 'Primär'].shape[0]
    print(f'Excluded {n_secondary} secondary transport patients')

    # adult secondary transport patients
    n_adult_secondary = data_df[(data_df['Einsatzart'] != 'Primär') & (data_df["Alter "] >= 16)].shape[0]
    print(f'Excluded {n_adult_secondary} adult secondary transport patients')
    # pediatric secondary transport patients
    n_pediatric_secondary = data_df[(data_df['Einsatzart'] != 'Primär') & (data_df["Alter "] < 16)].shape[0]
    print(f'Excluded {n_pediatric_secondary} pediatric secondary transport patients')
    data_df = data_df[data_df['Einsatzart'] == 'Primär']

adult_df = data_df[data_df["Alter "] >= 16]


