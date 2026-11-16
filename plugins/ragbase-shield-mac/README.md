# RAGbase Shield pour Claude (Mac)

**Claude travaille sur vos dossiers clients sans voir les vrais noms.**

[Français](#français) · [English](#english) · [Technical details](#technical-details)

## Français

### Ce que fait Shield

- Sur votre Mac, Shield remplace les noms et les autres données privées de vos documents par des codes, comme `PERSONNE_A` ou `SOCIETE_B`.
- Claude lit et rédige avec ces codes. Il ne voit pas les vrais noms.
- Le document final revient sur votre Mac, avec les vrais noms.

Gratuit et open source. Rien d'autre à installer.

### Ajouter Shield à Claude

Il vous faut l'application Claude pour Mac.

- **Le plus simple :** sur [shield.ragbase.ai](https://shield.ragbase.ai), cliquez sur le bouton pour Mac. L'application Claude s'ouvre. Cliquez sur **Add** à droite de « RAGbase Shield (Mac) ».
- **Autre façon :** dans l'application Claude, ouvrez **Customize > Plugins > Add > Add marketplace > Add from a repository**. Écrivez `ragbase/ragbase-shield-mac` et validez. Dans l'onglet **Discover**, cliquez ensuite sur **Add** à droite de « RAGbase Shield (Mac) ».

### Utiliser Shield en 3 étapes

1. Dans une conversation, écrivez : « Protège un dossier avec Shield ». Une fenêtre de votre Mac s'ouvre : choisissez le dossier du client. Claude reçoit seulement un code, par exemple `DOSSIER_1`. La première fois, votre Mac peut vous demander d'autoriser l'accès au dossier : cliquez sur **Autoriser**.
2. Demandez votre travail, par exemple : « Fais une synthèse du dossier » ou « Rédige une mise en demeure ». Claude lit les documents avec les codes.
3. Écrivez : « Enregistre le document final avec Shield ». Shield remet les vrais noms sur votre Mac. Le document est dans le dossier « Shield - sorties » de votre dossier client.

### Ce qui est protégé

Dans les documents du dossier choisi (Word, PDF avec du texte, Excel, texte, e-mails, pages web) : les noms de personnes et de sociétés, les adresses, les téléphones, les e-mails, les dates et lieux de naissance, les IBAN, les numéros de carte, de sécurité sociale, SIREN, SIRET et TVA, les références de procédure. Les noms des fichiers et des sous-dossiers sont protégés aussi.

**Attention :** un texte que vous tapez ou un fichier que vous déposez directement dans la conversation n'est pas masqué par Shield. Il part tel quel.

La détection n'est jamais parfaite : relisez les passages sensibles. Un PDF scanné (une image sans texte) ne peut pas être lu.

### Où restent vos données

- Vos documents d'origine et les vrais noms restent sur votre Mac. Claude ne reçoit que le texte avec les codes. Shield n'utilise pas internet.
- Le lien entre les codes et les vrais noms est gardé dans un coffre chiffré, sur votre Mac : le dossier « RAGbase Shield » de la Bibliothèque de votre compte (`~/Library/Application Support/RAGbase Shield`).
- La clé du coffre est dans le trousseau de votre session (application Trousseaux d'accès), sous le nom « ragbase-shield-plugin ».

### Retirer Shield

1. Dans l'application Claude, ouvrez **Customize > Plugins**, puis retirez « RAGbase Shield (Mac) ».
2. Pour effacer aussi le coffre : supprimez le dossier `~/Library/Application Support/RAGbase Shield` et l'élément « ragbase-shield-plugin » du trousseau de votre session (application Trousseaux d'accès).

Vos dossiers clients et les documents de « Shield - sorties » restent à leur place.

## English

### What Shield does

- On your Mac, Shield replaces names and other private data in your documents with codes such as `PERSONNE_A` or `SOCIETE_B`.
- Claude reads and writes with these codes. It does not see the real names.
- The final document comes back to your Mac with the real names.

Free and open source. Nothing else to install.

### Add Shield to Claude

You need the Claude app for Mac.

- **Easiest:** on [shield.ragbase.ai](https://shield.ragbase.ai), click the button for Mac. The Claude app opens. Click **Add** next to "RAGbase Shield (Mac)".
- **Other way:** in the Claude app, open **Customize > Plugins > Add > Add marketplace > Add from a repository**. Type `ragbase/ragbase-shield-mac` and confirm. Then, in the **Discover** tab, click **Add** next to "RAGbase Shield (Mac)".

### Use Shield in 3 steps

1. In a conversation, write: "Protège un dossier avec Shield" (or "Protect a folder with Shield"). A window of your Mac opens: choose the client folder. Claude only gets a code, for example `DOSSIER_1`. The first time, your Mac may ask you to allow access to the folder: click **Allow**.
2. Ask for your work, for example: "Summarise the file" or "Draft a formal notice". Claude reads the documents with the codes.
3. Write: "Save the final document with Shield". Shield puts the real names back on your Mac. The document is in the "Shield - sorties" folder of your client folder.

### What is protected

In the documents of the chosen folder (Word, PDF with text, Excel, text, e-mails, web pages): names of people and companies, addresses, phone numbers, e-mail addresses, dates and places of birth, IBANs, card, social security, SIREN, SIRET and VAT numbers, case references. File and sub-folder names are protected too.

**Note:** text you type, or a file you drop, directly into the conversation is not masked by Shield. It is sent as it is.

Detection is never perfect: check sensitive passages. A scanned PDF (an image with no text) cannot be read.

### Where your data stays

- Your original documents and the real names stay on your Mac. Claude only gets the text with the codes. Shield does not use the internet.
- The link between codes and real names is kept in an encrypted vault on your Mac: the "RAGbase Shield" folder in the Library of your account (`~/Library/Application Support/RAGbase Shield`).
- The vault key is in your login keychain (Keychain Access app), under the name "ragbase-shield-plugin".

### Remove Shield

1. In the Claude app, open **Customize > Plugins** and remove "RAGbase Shield (Mac)".
2. To also erase the vault: delete the folder `~/Library/Application Support/RAGbase Shield` and the item "ragbase-shield-plugin" in your login keychain (Keychain Access app).

Your client folders and the documents in "Shield - sorties" stay where they are.

## Technical details

- **Engine source:** [github.com/ragbase/ragbase-shield](https://github.com/ragbase/ragbase-shield), branch `plugin-claude-tab`, commit `03bd4816f0c32ddd6359cf39add300628c4c3d9a`. The plugin runs the module `ragbase_shield.claude_tab` (RAGbase Shield 0.4.2): seven MCP tools in French, detection rules only (no model).
- **Start:** `.mcp.json` runs `/bin/sh ${CLAUDE_PLUGIN_ROOT}/engine/shield` (no execute bit needed). The script starts the bundled Python with an empty environment (`env -i`) and `-I -B`.
- **Runtime:** Python 3.12.15, universal (Intel and Apple silicon), from python-build-standalone release 20261021 (PSF-2.0), in `engine/macos/`. The standard library is compiled in `engine/macos/lib/python312.zip`. The interpreter file is `engine/macos/bin/RAGbase Shield` (Python under the product's name: macOS shows this name when it asks for access to a folder).
- **No network:** nothing is downloaded at install or at run time, and the engine opens no network connection. The Mac build is tested with every network access denied by the system.
- **No hooks:** the plugin has no hooks. It can never block or change a message you send to Claude.
- **Integrity:** `SHA256SUMS` at the root of this repository gives the SHA-256 of every file of `plugins/ragbase-shield-mac/`. Check with `shasum -a 256 -c SHA256SUMS` (Mac) or `Get-FileHash` (Windows).
- **Licences:** RAGbase Shield is under the Apache License 2.0 (`LICENSE`, `NOTICE`). The runtime and the packages keep their own licences: full texts in `THIRD_PARTY_LICENSES` (also inside the plugin folder).

Bundled Python packages (39):

| Package | Version | Licence |
|---|---|---|
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| attrs | 26.1.0 | MIT |
| cffi | 2.1.1 | MIT-0 |
| charset-normalizer | 3.5.2 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| cryptography | 48.0.1 | Apache-2.0 OR BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| httpcore2 | 2.16.0 | BSD-3-Clause |
| httpx2 | 2.16.0 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| jaraco.classes | 3.4.0 | MIT License |
| jaraco.context | 6.1.2 | MIT |
| jaraco.functools | 4.6.0 | MIT |
| jsonschema | 4.26.0 | MIT |
| jsonschema-specifications | 2025.9.1 | MIT |
| keyring | 25.7.0 | MIT |
| lxml | 6.1.3 | BSD-3-Clause |
| mcp | 2.3.0 | MIT |
| mcp-types | 2.3.0 | MIT |
| more-itertools | 11.2.1 | MIT |
| opentelemetry-api | 1.45.1 | Apache-2.0 |
| pdfminer.six | 20260107 | MIT |
| pdfplumber | 0.11.10 | MIT License |
| pycparser | 3.12 | BSD-3-Clause |
| pydantic | 2.14.1 | MIT |
| pydantic_core | 2.50.1 | MIT |
| PyJWT | 2.15.1 | MIT |
| pypdf | 6.21.2 | BSD-3-Clause |
| python-docx | 1.2.0 | MIT |
| python-multipart | 0.0.32 | Apache-2.0 |
| referencing | 0.37.0 | MIT |
| rpds-py | 2026.9.1 | MIT |
| sse-starlette | 3.6.0 | BSD-3-Clause |
| starlette | 1.10.0 | BSD-3-Clause |
| truststore | 0.10.5 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| typing-inspection | 0.4.4 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |
