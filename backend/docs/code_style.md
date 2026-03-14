# Backend Code Style

`backend/main.py` und `backend/config.py` sind die stilistischen Referenzen fuer neuen Backend-Code.

Der Standard ist bewusst schlicht:

- kurze, direkte Module
- klare lineare Lesbarkeit
- wenig Abstraktion
- wenige Hilfsfunktionen
- keine unnoetigen neuen Dateien

## Grundprinzipien

- Bevorzuge einfache, lineare Kontrollfluesse statt cleverer Abstraktionen.
- Halte Module kompakt und funktional gut scanbar.
- Extrahiere nur dann Hilfsfunktionen oder neue Dateien, wenn Wiederverwendung oder Verstaendlichkeit dadurch real besser wird.
- Nutze klare, absolute Imports wie in `backend/main.py`.
- Benenne Dinge direkt und technisch praezise.
- Halte Konfiguration, Verdrahtung und abgeleitete Properties sichtbar und nah beieinander.

## Struktur im Stil von `backend/main.py`

- Imports stehen gesammelt und sauber gruppiert am Anfang.
- Groebere Abschnitte werden mit einfachen Trenner-Kommentaren markiert.
- Wichtige Initialisierung laeuft sichtbar von oben nach unten.
- Konstruktion und Verdrahtung von Komponenten bleiben explizit statt indirekt.
- Kurze erklaerende Kommentare sind gut, wenn sie Kontext geben. Kommentarflut ist nicht gewuenscht.

## Struktur im Stil von `backend/config.py`

- Klassen mit vielen Feldern werden fachlich in klare Abschnitte gruppiert.
- Defaults stehen direkt am Feld und nicht in versteckten Fabriken oder Mapping-Tabellen.
- Kleine abgeleitete Properties bleiben direkt bei den Feldern, auf die sie sich beziehen.
- Konfigurationsdateien bleiben deklarativ und ruhig lesbar, nicht meta-programmiert.
- Einfache Modulkonstanten wie `BACKEND_DIR` sind gut, wenn sie den Rest des Moduls klarer machen.

## Was bevorzugt ist

- flache Funktionen
- klarer Setup-Code
- explizite Zuweisungen
- deklarative Klassen mit sauber gruppierten Feldern
- kleine, gut erkennbare Datenfluesse
- gut lesbare Logging-Aufrufe
- mehr Klarheit als DRY um jeden Preis

## Was vermieden werden soll

- tiefe Utility-Schichten
- viele Mini-Hilfsfunktionen ohne echten Gewinn
- verstreute Ein-Zeilen-Abstraktionen
- unnötig generische Basisklassen oder Framework-Magie
- versteckte Seiteneffekte
- Regex, wenn ein solides Python-Paket die Aufgabe besser und lesbarer loest

## Praktische Regeln

- Wenn ein Modul beim Lesen in einem Zug verstanden werden kann, ist das gut.
- Wenn eine Extraktion nur zwei Zeilen spart, bleibt sie meist im Modul.
- Wenn ein Kommentar nur den Code wiederholt, faellt er weg.
- Wenn ein Package ein Problem sauber loest, ist das besser als eigene Parser- oder Regex-Bastelei.
- Neue Dateien brauchen einen klaren funktionalen Grund.
- Settings-, API- und Verdrahtungsmodule sollen eher geordnet als generisch wirken.
- Abschnittskommentare sollen Orientierung geben, nicht den Code verdoppeln.
- Properties und kleine Helfer sollen lokal bleiben und nicht in Utility-Module auswandern.

## Kurzform

Schreibe neuen Backend-Code so, dass er wie eine ruhige, gut lesbare Verdrahtung oder deklarative Konfiguration aussieht und nicht wie ein kleines Framework.
