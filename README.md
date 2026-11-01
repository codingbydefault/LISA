# Lisa — Limited Incorporated

Lisa is een lokale chatapp met Ollama. Je kunt eigen bots maken zoals custom GPTs: geef elke bot een naam en kleur en stel met de functieprompt bij het aanmaken in hoe die bot werkt. Daarna kies je een bot en chat je rechtstreeks met die persona. Lisa kan desgewenst het antwoord van een andere bot controleren. De functieprompt past de instructies aan; het traint of fine-tunet de modelgewichten niet.

Botprofielen worden lokaal in de browser opgeslagen. Elk botgesprek is apart en blijft alleen zolang de pagina open is; verversen of sluiten wist de gesprekken. Geüploade kennisbestanden worden lokaal bewaard in `knowledge/` totdat je ze zelf verwijdert. Onderzoek- en salesbots hebben geen webtoegang. E-mails worden niet gelezen of verzonden.

## Benodigd

- Python 3.10 of nieuwer
- [Ollama](https://ollama.com/) geïnstalleerd en actief
- `faster-whisper` voor lokale spraakherkenning
- Een lokaal Ollama-model

Download bijvoorbeeld een compact model:

```sh
ollama pull qwen2.5:3b
```

Welk model prettig draait, hangt af van het beschikbare geheugen en de processor/GPU. Kies een kleiner model bij weinig geheugen; installeer grotere modellen alleen als je computer die aankan.

## Installeren en starten

In Windows PowerShell:

```sh
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

In Linux of macOS:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open daarna <http://127.0.0.1:8765>. De app bindt standaard alleen aan localhost. Ollama moet bereikbaar zijn op `http://127.0.0.1:11434`.

De eerste spraaktranscriptie downloadt het Whisper `base`-model als het nog niet lokaal beschikbaar is. Voor een spraakbericht klik je op de microfoon, spreek je je vraag in en klik je opnieuw om te stoppen. De herkende tekst verschijnt eerst in het invoerveld zodat je die kunt controleren voordat je hem verstuurt. Audio wordt lokaal verwerkt en niet naar schijf geschreven. Je kunt antwoorden optioneel automatisch laten voorlezen; daarvoor gebruikt de browser een geïnstalleerde stem, waarvan beschikbaarheid en verwerking afhangen van browser en besturingssysteem. Transcript en samenvatting voor gespreksnotities blijven alleen in de open browserpagina; sluiten of verversen wist ze. De app luistert niet op de achtergrond: start en stop elke opnamesessie zelf, en neem gesprekken alleen op nadat alle betrokkenen toestemming hebben gegeven. Gebruik voor notities de momentknop, selecteer transcriptiepassages en vraag daarna een samenvatting. De geselecteerde transcriptie gaat alleen naar je lokale Ollama-installatie om samen te vatten.

Optioneel kun je de host, poort of Ollama-URL instellen met `HOST`, `PORT` en `OLLAMA_URL`. Deel de app niet op je netwerk zonder eerst passende authenticatie en toegangsbeveiliging toe te voegen.

## Eigen kennis

Voeg via de interface `.txt`- of `.md`-bestanden toe. De bestanden worden lokaal bewaard in `knowledge/`. De app zoekt eenvoudige tekstfragmenten op basis van overeenkomende woorden en stuurt relevante passages samen met je vraag naar Ollama. Dit is een lichte eerste versie; het is geen fine-tuning en de documentzoekfunctie is geen semantische vectorzoekmachine.

## Later uitbreiden

- Gmail-lezen en concepten maken via Google OAuth, met handmatige goedkeuring vóór verzending.
- Webonderzoek voor leads met bronvermelding en controle op toepasselijke privacy- en anti-spamregels.
- Betere documentextractie en semantisch zoeken.
- Een passende lokale codeer- of redeneermodelfamilie kiezen op basis van je hardware.
