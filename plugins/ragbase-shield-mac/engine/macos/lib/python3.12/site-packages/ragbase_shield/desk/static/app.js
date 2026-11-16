/*
 * RAGbase Shield · Shield Desk UI
 * Plain JavaScript, no dependencies, no external requests, works offline.
 *
 * This page runs on the user's own computer and talks only to the local Shield Desk
 * JSON API (ARCHITECTURE.md section 6). It is a local page, so it may show real values;
 * every real value is inserted as text (textContent), never parsed as HTML.
 * Nothing the user types is stored in the browser. Only the session token
 * (sessionStorage), the language choice and the last dossier CODE used (localStorage) are kept.
 */
(() => {
  "use strict";

  /* ------------------------------------------------------------ token */

  const TOKEN_KEY = "ragbase-shield.desk.token";
  const LANG_KEY = "ragbase-shield.desk.lang";
  const DOSSIER_KEY = "ragbase-shield.desk.dossier"; // a code such as DOSSIER_2, never a title or a path
  let token = "";

  /** Read ?t= once, keep it for this tab, then remove it from the address bar. */
  function captureToken() {
    let url;
    try {
      url = new URL(window.location.href);
    } catch (e) {
      return;
    }
    const fromUrl = url.searchParams.get("t");
    if (fromUrl) {
      token = fromUrl;
      try {
        window.sessionStorage.setItem(TOKEN_KEY, fromUrl);
      } catch (e) {
        /* storage blocked: the token stays in memory for this page */
      }
      url.searchParams.delete("t");
      const query = url.searchParams.toString();
      try {
        window.history.replaceState(window.history.state, "", url.pathname + (query ? "?" + query : "") + url.hash);
      } catch (e) {
        /* the page still works */
      }
      return;
    }
    try {
      token = window.sessionStorage.getItem(TOKEN_KEY) || "";
    } catch (e) {
      token = "";
    }
  }
  captureToken();

  /* ------------------------------------------------------------ constants */

  const TABS = ["dossiers", "masquer", "restaurer", "fichiers", "journal"];
  const KINDS = ["PERSON", "ORG", "ADDRESS", "EMAIL", "PHONE", "IBAN", "NIR", "SIREN", "SIRET",
    "VAT", "CARD", "CASE_REF", "ID_CODE", "URL", "BIRTHDATE", "USERNAME", "EMPLOYEE_ID", "ACCOUNT_NUMBER"];
  const PREFIX_KIND = {
    PERSONNE: "PERSON", PERSON: "PERSON", SOCIETE: "ORG", COMPANY: "ORG",
    ADRESSE: "ADDRESS", ADDRESS: "ADDRESS", EMAIL: "EMAIL", TEL: "PHONE", PHONE: "PHONE",
    IBAN: "IBAN", NIR: "NIR", SSN: "NIR", SIREN: "SIREN", SIRET: "SIRET", TVA: "VAT", VAT: "VAT",
    CARTE: "CARD", CARD: "CARD", RG: "CASE_REF", CASEREF: "CASE_REF", IDENT: "ID_CODE",
    SITE: "URL", NAISSANCE: "BIRTHDATE", BIRTHDATE: "BIRTHDATE",
    PSEUDO: "USERNAME", USERNAME: "USERNAME", MATRICULE: "EMPLOYEE_ID", EMPLOYEEID: "EMPLOYEE_ID",
    LIEU: "BIRTHPLACE", PLACE: "BIRTHPLACE",
    COMPTE: "ACCOUNT_NUMBER", ACCOUNT: "ACCOUNT_NUMBER",
  };
  // Same codes as engine/types.py CODE_RE (\b is equivalent to [A-Za-z0-9_] boundaries here).
  const CODE_RE = /\b(?:(?:PERSONNE|PERSON|SOCIETE|COMPANY)_[A-Z]{1,3}|(?:ADRESSE|ADDRESS|EMAIL|TEL|PHONE|IBAN|NIR|SSN|SIREN|SIRET|TVA|VAT|CARTE|CARD|RG|CASEREF|IDENT|SITE|NAISSANCE|BIRTHDATE|PSEUDO|USERNAME|MATRICULE|EMPLOYEEID|COMPTE|ACCOUNT|LIEU|PLACE)_[0-9]{1,4}|DOSSIER_[0-9]{1,4}|DOC_[0-9]{1,5}|OUTPUT_[0-9]{1,5})\b/g;
  const PREVIEW_CHARS = 50000;
  // Turn off spell checkers and writing assistants that send text to a server.
  const PRIVATE = {
    autocomplete: "off", autocapitalize: "off", autocorrect: "off", spellcheck: "false",
    "data-gramm": "false", "data-gramm_editor": "false", "data-enable-grammarly": "false", "data-lt-active": "false",
  };

  /* ------------------------------------------------------------ strings (one dictionary) */

  const I18N = {
    fr: {
      doc_title: "Shield Desk · RAGbase Shield",
      skip: "Aller au contenu",
      lang_group: "Langue",
      tabs_label: "Sections de Shield Desk",
      tab_dossiers: "Dossiers",
      tab_masquer: "Masquer",
      tab_restaurer: "Restaurer",
      tab_fichiers: "Fichiers",
      tab_journal: "Journal",
      conn_checking: "Connexion…",
      conn_ok: "Connecté",
      conn_ok_version: "Connecté · v{v}",
      conn_off: "Hors ligne",
      conn_token: "Jeton refusé",
      retry: "Réessayer",
      close: "Fermer",
      refresh: "Actualiser",
      loading: "Chargement…",
      clear_btn: "Effacer",
      copy_failed: "Copie impossible. Sélectionnez le texte puis copiez-le avec Ctrl+C (Cmd+C sur Mac).",
      dos_add_title: "Ajouter un dossier",
      dos_add_intro: "Indiquez le dossier client qui se trouve sur cet ordinateur. Shield lit les documents sur place et prépare un code pour chaque nom et chaque identifiant.",
      dos_warn_attach: "Connectez un dossier, ne joignez pas le fichier à la conversation. Un fichier joint ou un texte collé dans le chat part chez le fournisseur avant que Shield puisse agir.",
      dos_folder_label: "Chemin du dossier",
      dos_folder_hint: "Chemin complet, par exemple /Users/vous/Dossiers/Client 2026 ou C:\\Dossiers\\Client 2026.",
      dos_folder_empty: "Indiquez d'abord le chemin du dossier.",
      dos_title_label: "Titre (facultatif)",
      dos_title_hint: "Visible seulement sur cette page. Les assistants IA ne voient que le code du dossier, par exemple DOSSIER_1.",
      dos_add_btn: "Ajouter et analyser",
      dos_adding: "Ajout en cours…",
      dos_added: "{code} ajouté.",
      dos_list_title: "Vos dossiers",
      dos_empty: "Aucun dossier pour l'instant. Ajoutez le premier ci-dessus.",
      dos_locked: "La liste des dossiers s'affiche quand la page est ouverte avec son lien privé (jeton). Relancez « uvx ragbase-shield desk » ou utilisez le lien affiché dans le terminal.",
      dos_offline: "Shield Desk ne répond pas : la liste des dossiers ne peut pas s'afficher.",
      dos_loading: "Chargement des dossiers…",
      dos_docs_one: "{n} document",
      dos_docs_other: "{n} documents",
      dos_ents_one: "{n} entité",
      dos_ents_other: "{n} entités",
      dos_show: "Voir les entités",
      dos_hide: "Replier",
      dos_scan: "Analyser le dossier",
      dos_scanning: "Analyse en cours…",
      dos_scan_done: "Analyse terminée : {summary}.",
      dos_scan_none: "Analyse terminée : aucune entité trouvée.",
      dos_remove: "Retirer",
      dos_remove_confirm: "Retirer {code} de Shield ? Vos fichiers restent en place, mais les codes de ce dossier ne pourront plus être restaurés.",
      dos_removed: "{code} retiré de Shield.",
      dos_counts_label: "Nombre d'entités par type",
      ent_filter_label: "Filtrer les entités",
      ent_filter_ph: "Code, valeur ou type",
      ent_col_code: "Code",
      ent_col_kind: "Type",
      ent_col_value: "Valeur réelle",
      ent_col_clear: "Garder en clair",
      ent_clear_aria: "Garder {code} en clair",
      ent_none: "Aucune entité. Lancez une analyse ou ajoutez un nom.",
      ent_no_match: "Aucune entité ne correspond au filtre.",
      ent_kept_on: "{code} sera gardé en clair.",
      ent_kept_off: "{code} sera de nouveau masqué.",
      add_name_title: "Ajouter un nom ou un terme",
      add_name_hint: "Pour un nom que Shield n'a pas repéré : client, partie adverse, nom de projet.",
      add_kind_label: "Type",
      add_value_label: "Valeur",
      add_value_ph: "ex. Hélène Marchand",
      add_btn: "Ajouter",
      add_empty: "Saisissez d'abord une valeur.",
      add_done: "Ajouté sous le code {code}.",
      dossier_label: "Dossier",
      dossier_none: "Sans dossier (codes temporaires)",
      dossier_pick: "Choisissez un dossier",
      mask_title: "Masquer un texte",
      mask_hint_none: "Aucun dossier choisi : les noms enregistrés deviennent [PERSONNE] ou [SOCIETE] et les codes sont temporaires. Vous ne pourrez pas restaurer les noms ensuite. Choisissez le dossier client.",
      mask_hint_dossier: "Les codes de ce dossier restent stables. Vous pourrez restaurer les noms dans l'onglet Restaurer.",
      mask_input_label: "Texte d'origine",
      mask_input_hint: "Ce texte reste sur cet ordinateur. Raccourci : Ctrl+Entrée (Cmd+Entrée sur Mac).",
      mask_btn: "Masquer",
      masking: "Masquage…",
      mask_empty: "Collez d'abord un texte à masquer.",
      mask_done_one: "Texte masqué : {n} remplacement.",
      mask_done_other: "Texte masqué : {n} remplacements.",
      mask_result_title: "Texte masqué",
      mask_stale: "Le texte d'origine ou les réglages du dossier ont changé. Cliquez sur Masquer pour mettre à jour.",
      mask_reminder: "Collez ce texte dans Claude, ChatGPT ou Copilot. Ne collez jamais le texte d'origine.",
      mask_copy_btn: "Copier le texte masqué",
      mask_copied: "Texte masqué copié. Collez-le dans votre assistant IA.",
      review_title: "Vérifiez avant de copier",
      review_summary_one: "{n} remplacement : {summary}.",
      review_summary_other: "{n} remplacements : {summary}.",
      review_none: "Aucune donnée détectée. Relisez le texte avant de le copier.",
      review_kept: "gardé en clair",
      review_nodossier: "Sans dossier, la vérification montre seulement les codes. Choisissez un dossier pour voir les vraies valeurs, garder un terme en clair ou ajouter un nom oublié.",
      review_add_title: "Un nom n'a pas été masqué ?",
      review_add_btn: "Ajouter et masquer à nouveau",
      review_table_label: "Codes trouvés",
      restore_title: "Restaurer les noms",
      restore_intro: "Collez la réponse de l'assistant IA. Shield remplace les codes par les vrais noms, sur cet ordinateur.",
      restore_hint_none: "Aucun dossier. Un texte masqué sans dossier ne peut pas être restauré.",
      restore_hint_pick: "Choisissez le dossier qui a servi à masquer le texte.",
      restore_input_label: "Réponse de l'IA (avec les codes)",
      restore_input_hint: "Raccourci : Ctrl+Entrée (Cmd+Entrée sur Mac).",
      restore_btn: "Restaurer les noms",
      restoring: "Restauration…",
      restore_need_dossier: "Choisissez d'abord le dossier.",
      restore_empty: "Collez d'abord la réponse de l'IA.",
      restore_result_title: "Texte restauré",
      restore_count_one: "{n} code restauré.",
      restore_count_other: "{n} codes restaurés.",
      restore_unknown_title: "Codes inconnus dans ce dossier",
      restore_unknown_text: "L'assistant IA a peut-être inventé ou modifié ces codes. Ils restent tels quels dans le texte : vérifiez-les avant de l'utiliser.",
      restore_note: "Ce texte contient les vrais noms. Gardez-le sur cet ordinateur : ne le collez pas dans le chat.",
      restore_copy_btn: "Copier le texte restauré",
      restore_copied: "Texte restauré copié.",
      files_title: "Masquer des fichiers",
      files_intro: "Shield Desk lit les fichiers sur cet ordinateur et vous rend une copie masquée en texte brut.",
      files_hint_none: "Aucun dossier choisi : les noms enregistrés deviennent [PERSONNE] ou [SOCIETE] et les codes sont temporaires. Vous ne pourrez pas restaurer les noms ensuite. Choisissez le dossier client.",
      files_hint_dossier: "Les codes suivent ceux du dossier choisi.",
      files_drop: "Déposez vos fichiers ici",
      files_or: "ou",
      files_pick: "Choisir des fichiers",
      files_types: "Formats : PDF, DOCX, TXT, MD, EML, HTML.",
      files_sending_one: "Masquage de {n} fichier…",
      files_sending_other: "Masquage de {n} fichiers…",
      files_done_one: "{n} fichier masqué.",
      files_done_other: "{n} fichiers masqués.",
      files_results_title: "Fichiers masqués",
      files_clear: "Tout effacer",
      files_reminder: "Utilisez seulement la copie masquée dans le chat, jamais le fichier d'origine.",
      files_original: "Fichier d'origine",
      files_masked_name: "Nom masqué",
      files_download: "Télécharger (.txt)",
      files_copy: "Copier",
      files_copied: "Texte masqué copié.",
      files_preview: "Aperçu du texte masqué",
      files_preview_cut: "Aperçu limité aux {n} premiers caractères. Le téléchargement contient tout le texte.",
      files_no_text: "Aucun texte lisible. S'il s'agit d'un PDF scanné, la reconnaissance de texte (OCR) n'est pas disponible en v0.1.",
      files_default_name: "texte-masque",
      journal_title: "Journal",
      journal_intro: "Le journal garde la trace des opérations avec des codes et des nombres seulement, jamais de vraies valeurs.",
      journal_filter: "Dossier",
      journal_all: "Tous les dossiers",
      journal_col_time: "Date",
      journal_col_event: "Événement",
      journal_col_dossier: "Dossier",
      journal_col_counts: "Détails",
      journal_empty: "Aucune opération enregistrée.",
      ev_mask: "Masquage",
      ev_restore: "Restauration",
      ev_scan: "Analyse",
      ev_add_dossier: "Ajout d'un dossier",
      ev_remove_dossier: "Retrait d'un dossier",
      ev_register: "Ajout d'un nom",
      ev_keep_clear: "Garder en clair",
      ev_read_document: "Lecture d'un document",
      ev_list_documents: "Liste des documents",
      ev_search: "Recherche",
      ev_save_draft: "Brouillon enregistré",
      ev_files: "Fichiers masqués",
      ev_export: "Export du dossier",
      ev_import: "Import du dossier",
      ev_dossier_added: "Dossier ajouté",
      ev_dossier_removed: "Dossier retiré",
      ev_dossier_imported: "Dossier importé",
      ev_hook_prompt_blocked: "Message bloqué (Claude Code)",
      ev_hook_read_denied: "Lecture directe refusée (Claude Code)",
      ev_hook_output_masked: "Résultat d'outil masqué (Claude Code)",
      count_documents: "Documents",
      count_entities: "Entités",
      count_files: "Fichiers",
      count_chars: "Caractères",
      count_codes: "Codes",
      count_restored: "Restaurés",
      count_unknown: "Codes inconnus",
      count_unknown_codes: "Codes inconnus",
      count_spans: "Éléments masqués",
      count_new_entities: "Nouvelles entités",
      count_parties: "Parties",
      count_results: "Résultats",
      count_errors: "Erreurs",
      count_values: "Valeurs",
      count_bytes: "Octets",
      count_format: "Format",
      count_output: "Sortie",
      count_document: "Document",
      count_code: "Code",
      kind_PERSON: "Personnes",
      kind_ORG: "Sociétés",
      kind_ADDRESS: "Adresses",
      kind_EMAIL: "E-mails",
      kind_PHONE: "Téléphones",
      kind_IBAN: "IBAN",
      kind_NIR: "N° de sécurité sociale",
      kind_SIREN: "SIREN",
      kind_SIRET: "SIRET",
      kind_VAT: "N° de TVA",
      kind_CARD: "Cartes bancaires",
      kind_CASE_REF: "Références d'affaires",
      kind_ID_CODE: "Autres identifiants",
      kind_URL: "Sites web",
      kind_BIRTHDATE: "Dates de naissance",
      kind_USERNAME: "Pseudos et identifiants de connexion",
      kind_EMPLOYEE_ID: "Matricules",
      kind_ACCOUNT_NUMBER: "Numéros de compte",
      kind1_PERSON: "Personne",
      kind1_ORG: "Société",
      kind1_ADDRESS: "Adresse",
      kind1_EMAIL: "E-mail",
      kind1_PHONE: "Téléphone",
      kind1_IBAN: "IBAN",
      kind1_NIR: "N° de sécurité sociale",
      kind1_SIREN: "SIREN",
      kind1_SIRET: "SIRET",
      kind1_VAT: "N° de TVA",
      kind1_CARD: "Carte bancaire",
      kind1_CASE_REF: "Référence d'affaire (RG, pourvoi)",
      kind1_ID_CODE: "Autre identifiant",
      kind1_URL: "Site web",
      kind1_BIRTHDATE: "Date de naissance",
      kind1_USERNAME: "Pseudo",
      kind1_EMPLOYEE_ID: "Matricule",
      kind1_ACCOUNT_NUMBER: "Numéro de compte",
      ext_title: "Extension navigateur",
      ext_intro: "L'extension RAGbase Shield pour Chrome et Edge masque le texte que vous collez dans chatgpt.com, claude.ai, copilot.microsoft.com, m365.cloud.microsoft et gemini.google.com. Dans ses options, indiquez le port de cette adresse (7878 par défaut) et le jeton ci-dessous.",
      ext_address: "Adresse de Shield Desk",
      ext_token: "Jeton",
      ext_show: "Afficher",
      ext_hide: "Cacher",
      ext_copy: "Copier le jeton",
      ext_copied: "Jeton copié. Collez-le dans les options de l'extension.",
      ext_no_token: "Aucun jeton dans cette page. Relancez « uvx ragbase-shield desk ».",
      ext_secret: "Ce jeton protège Shield Desk. Ne le partagez pas et ne le collez jamais dans un chat.",
      footer_privacy: "Pseudonymisation locale et réversible. Rien n'est envoyé à RAGbase.",
      footer_limits: "Shield réduit le risque sans le supprimer : relisez toujours le texte masqué avant de le copier.",
      footer_more: "Pour aller plus loin : RAGbase",
      new_tab: " (s'ouvre dans un nouvel onglet)",
      err_token: "Le jeton est manquant ou invalide. Relancez « uvx ragbase-shield desk », ou utilisez le lien affiché dans le terminal : il contient le jeton.",
      err_offline: "Shield Desk ne répond pas. Vérifiez que « uvx ragbase-shield desk » tourne toujours, puis réessayez.",
      err_notfound: "Élément introuvable. Le dossier a peut-être été retiré : actualisez la liste.",
      err_too_big: "Fichier trop volumineux pour Shield Desk.",
      err_bad_request: "Requête refusée : {msg}",
      err_server: "Erreur de Shield Desk (code {status}). {msg}",
      err_bad_response: "Réponse inattendue de Shield Desk.",
      err_generic: "Une erreur est survenue.",
    },
    en: {
      doc_title: "Shield Desk · RAGbase Shield",
      skip: "Skip to content",
      lang_group: "Language",
      tabs_label: "Shield Desk sections",
      tab_dossiers: "Matters",
      tab_masquer: "Mask",
      tab_restaurer: "Restore",
      tab_fichiers: "Files",
      tab_journal: "Log",
      conn_checking: "Connecting…",
      conn_ok: "Connected",
      conn_ok_version: "Connected · v{v}",
      conn_off: "Offline",
      conn_token: "Token refused",
      retry: "Try again",
      close: "Close",
      refresh: "Refresh",
      loading: "Loading…",
      clear_btn: "Clear",
      copy_failed: "Cannot copy. Select the text, then copy it with Ctrl+C (Cmd+C on Mac).",
      dos_add_title: "Add a matter",
      dos_add_intro: "Point Shield to the client folder on this computer. Shield reads the documents where they are and prepares a code for each name and identifier.",
      dos_warn_attach: "Connect a folder, do not attach the file to the conversation. An attached file or text pasted into the chat reaches the provider before Shield can act.",
      dos_folder_label: "Folder path",
      dos_folder_hint: "Full path, for example /Users/you/Matters/Client 2026 or C:\\Matters\\Client 2026.",
      dos_folder_empty: "Enter the folder path first.",
      dos_title_label: "Title (optional)",
      dos_title_hint: "Shown on this page only. AI assistants only see the matter code, for example DOSSIER_1.",
      dos_add_btn: "Add and scan",
      dos_adding: "Adding…",
      dos_added: "{code} added.",
      dos_list_title: "Your matters",
      dos_empty: "No matter yet. Add the first one above.",
      dos_locked: "The list of matters shows when the page is opened with its private link (token). Run « uvx ragbase-shield desk » again or use the link shown in the terminal.",
      dos_offline: "Shield Desk does not answer: the list of matters cannot be shown.",
      dos_loading: "Loading the matters…",
      dos_docs_one: "{n} document",
      dos_docs_other: "{n} documents",
      dos_ents_one: "{n} entity",
      dos_ents_other: "{n} entities",
      dos_show: "Show entities",
      dos_hide: "Collapse",
      dos_scan: "Scan the folder",
      dos_scanning: "Scanning…",
      dos_scan_done: "Scan done: {summary}.",
      dos_scan_none: "Scan done: no entity found.",
      dos_remove: "Remove",
      dos_remove_confirm: "Remove {code} from Shield? Your files stay where they are, but the codes of this matter can no longer be restored.",
      dos_removed: "{code} removed from Shield.",
      dos_counts_label: "Entities per type",
      ent_filter_label: "Filter entities",
      ent_filter_ph: "Code, value or type",
      ent_col_code: "Code",
      ent_col_kind: "Type",
      ent_col_value: "Real value",
      ent_col_clear: "Keep in clear",
      ent_clear_aria: "Keep {code} in clear",
      ent_none: "No entity. Run a scan or add a name.",
      ent_no_match: "No entity matches the filter.",
      ent_kept_on: "{code} will stay in clear.",
      ent_kept_off: "{code} will be masked again.",
      add_name_title: "Add a name or a term",
      add_name_hint: "For a name Shield did not catch: client, opposing party, project name.",
      add_kind_label: "Type",
      add_value_label: "Value",
      add_value_ph: "e.g. Hélène Marchand",
      add_btn: "Add",
      add_empty: "Enter a value first.",
      add_done: "Added as {code}.",
      dossier_label: "Matter",
      dossier_none: "No matter (temporary codes)",
      dossier_pick: "Choose a matter",
      mask_title: "Mask a text",
      mask_hint_none: "No matter chosen: registered names become [PERSON] or [COMPANY] and the codes are temporary. You cannot restore the names later. Choose the client matter.",
      mask_hint_dossier: "The codes of this matter stay the same. You can restore the names in the Restore tab.",
      mask_input_label: "Original text",
      mask_input_hint: "This text stays on this computer. Shortcut: Ctrl+Enter (Cmd+Enter on Mac).",
      mask_btn: "Mask",
      masking: "Masking…",
      mask_empty: "Paste a text to mask first.",
      mask_done_one: "Text masked: {n} replacement.",
      mask_done_other: "Text masked: {n} replacements.",
      mask_result_title: "Masked text",
      mask_stale: "The original text or the matter settings changed. Click Mask to update.",
      mask_reminder: "Paste this text into Claude, ChatGPT or Copilot. Never paste the original text.",
      mask_copy_btn: "Copy the masked text",
      mask_copied: "Masked text copied. Paste it into your AI assistant.",
      review_title: "Check before you copy",
      review_summary_one: "{n} replacement: {summary}.",
      review_summary_other: "{n} replacements: {summary}.",
      review_none: "No data detected. Read the text again before you copy it.",
      review_kept: "kept in clear",
      review_nodossier: "Without a matter, this check shows the codes only. Choose a matter to see the real values, keep a term in clear or add a missed name.",
      review_add_title: "A name was not masked?",
      review_add_btn: "Add and mask again",
      review_table_label: "Codes found",
      restore_title: "Restore the names",
      restore_intro: "Paste the AI assistant's answer. Shield replaces the codes with the real names, on this computer.",
      restore_hint_none: "No matter yet. A text masked without a matter cannot be restored.",
      restore_hint_pick: "Choose the matter that was used to mask the text.",
      restore_input_label: "AI answer (with codes)",
      restore_input_hint: "Shortcut: Ctrl+Enter (Cmd+Enter on Mac).",
      restore_btn: "Restore the names",
      restoring: "Restoring…",
      restore_need_dossier: "Choose the matter first.",
      restore_empty: "Paste the AI answer first.",
      restore_result_title: "Restored text",
      restore_count_one: "{n} code restored.",
      restore_count_other: "{n} codes restored.",
      restore_unknown_title: "Codes unknown in this matter",
      restore_unknown_text: "The AI assistant may have invented or changed these codes. They stay as they are in the text: check them before you use it.",
      restore_note: "This text contains the real names. Keep it on this computer: do not paste it into the chat.",
      restore_copy_btn: "Copy the restored text",
      restore_copied: "Restored text copied.",
      files_title: "Mask files",
      files_intro: "Shield Desk reads the files on this computer and gives you a masked plain-text copy.",
      files_hint_none: "No matter chosen: registered names become [PERSON] or [COMPANY] and the codes are temporary. You cannot restore the names later. Choose the client matter.",
      files_hint_dossier: "The codes follow the codes of the chosen matter.",
      files_drop: "Drop your files here",
      files_or: "or",
      files_pick: "Choose files",
      files_types: "Formats: PDF, DOCX, TXT, MD, EML, HTML.",
      files_sending_one: "Masking {n} file…",
      files_sending_other: "Masking {n} files…",
      files_done_one: "{n} file masked.",
      files_done_other: "{n} files masked.",
      files_results_title: "Masked files",
      files_clear: "Clear all",
      files_reminder: "Use only the masked copy in the chat, never the original file.",
      files_original: "Original file",
      files_masked_name: "Masked name",
      files_download: "Download (.txt)",
      files_copy: "Copy",
      files_copied: "Masked text copied.",
      files_preview: "Preview of the masked text",
      files_preview_cut: "Preview limited to the first {n} characters. The download contains the full text.",
      files_no_text: "No readable text. If this is a scanned PDF, text recognition (OCR) is not available in v0.1.",
      files_default_name: "masked-text",
      journal_title: "Log",
      journal_intro: "The log records operations with codes and numbers only, never real values.",
      journal_filter: "Matter",
      journal_all: "All matters",
      journal_col_time: "Date",
      journal_col_event: "Event",
      journal_col_dossier: "Matter",
      journal_col_counts: "Details",
      journal_empty: "No operation recorded.",
      ev_mask: "Mask",
      ev_restore: "Restore",
      ev_scan: "Scan",
      ev_add_dossier: "Matter added",
      ev_remove_dossier: "Matter removed",
      ev_register: "Name added",
      ev_keep_clear: "Keep in clear",
      ev_read_document: "Document read",
      ev_list_documents: "Document list",
      ev_search: "Search",
      ev_save_draft: "Draft saved",
      ev_files: "Files masked",
      ev_export: "Matter export",
      ev_import: "Matter import",
      ev_dossier_added: "Matter added",
      ev_dossier_removed: "Matter removed",
      ev_dossier_imported: "Matter imported",
      ev_hook_prompt_blocked: "Prompt blocked (Claude Code)",
      ev_hook_read_denied: "Direct read denied (Claude Code)",
      ev_hook_output_masked: "Tool result masked (Claude Code)",
      count_documents: "Documents",
      count_entities: "Entities",
      count_files: "Files",
      count_chars: "Characters",
      count_codes: "Codes",
      count_restored: "Restored",
      count_unknown: "Unknown codes",
      count_unknown_codes: "Unknown codes",
      count_spans: "Masked items",
      count_new_entities: "New entities",
      count_parties: "Parties",
      count_results: "Results",
      count_errors: "Errors",
      count_values: "Values",
      count_bytes: "Bytes",
      count_format: "Format",
      count_output: "Output",
      count_document: "Document",
      count_code: "Code",
      kind_PERSON: "People",
      kind_ORG: "Companies",
      kind_ADDRESS: "Addresses",
      kind_EMAIL: "Emails",
      kind_PHONE: "Phone numbers",
      kind_IBAN: "IBANs",
      kind_NIR: "Social security numbers",
      kind_SIREN: "SIREN",
      kind_SIRET: "SIRET",
      kind_VAT: "VAT numbers",
      kind_CARD: "Payment cards",
      kind_CASE_REF: "Case references",
      kind_ID_CODE: "Other identifiers",
      kind_URL: "Websites",
      kind_BIRTHDATE: "Birth dates",
      kind_USERNAME: "Usernames and handles",
      kind_EMPLOYEE_ID: "Employee IDs",
      kind_ACCOUNT_NUMBER: "Account numbers",
      kind1_PERSON: "Person",
      kind1_ORG: "Company",
      kind1_ADDRESS: "Address",
      kind1_EMAIL: "Email",
      kind1_PHONE: "Phone number",
      kind1_IBAN: "IBAN",
      kind1_NIR: "Social security number",
      kind1_SIREN: "SIREN",
      kind1_SIRET: "SIRET",
      kind1_VAT: "VAT number",
      kind1_CARD: "Payment card",
      kind1_CASE_REF: "Case reference (docket)",
      kind1_ID_CODE: "Other identifier",
      kind1_URL: "Website",
      kind1_BIRTHDATE: "Birth date",
      kind1_USERNAME: "Username",
      kind1_EMPLOYEE_ID: "Employee ID",
      kind1_ACCOUNT_NUMBER: "Account number",
      ext_title: "Browser extension",
      ext_intro: "The RAGbase Shield extension for Chrome and Edge masks the text you paste into chatgpt.com, claude.ai, copilot.microsoft.com, m365.cloud.microsoft and gemini.google.com. In its options, enter the port of this address (7878 by default) and the token below.",
      ext_address: "Shield Desk address",
      ext_token: "Token",
      ext_show: "Show",
      ext_hide: "Hide",
      ext_copy: "Copy the token",
      ext_copied: "Token copied. Paste it into the extension options.",
      ext_no_token: "No token in this page. Run « uvx ragbase-shield desk » again.",
      ext_secret: "This token protects Shield Desk. Do not share it and never paste it into a chat.",
      footer_privacy: "Local, reversible pseudonymisation. Nothing is sent to RAGbase.",
      footer_limits: "Shield lowers the risk, it does not remove it: always read the masked text before you copy it.",
      footer_more: "Go further: RAGbase",
      new_tab: " (opens in a new tab)",
      err_token: "The token is missing or invalid. Run « uvx ragbase-shield desk » again, or use the link shown in the terminal: it holds the token.",
      err_offline: "Shield Desk does not answer. Check that « uvx ragbase-shield desk » is still running, then try again.",
      err_notfound: "Item not found. The matter may have been removed: refresh the list.",
      err_too_big: "File too large for Shield Desk.",
      err_bad_request: "Request refused: {msg}",
      err_server: "Shield Desk error (code {status}). {msg}",
      err_bad_response: "Unexpected answer from Shield Desk.",
      err_generic: "Something went wrong.",
    },
  };

  /* ------------------------------------------------------------ state */

  const state = {
    lang: "fr",
    tab: "dossiers",
    conn: "checking",
    version: "",
    alert: null, // {key} or {err}
    dossiers: [],
    current: "", // dossier code shared by the Masquer, Restaurer and Fichiers selectors
    open: Object.create(null),
    entities: Object.create(null),
    filter: Object.create(null),
    scanMsg: Object.create(null),
    scanning: new Set(),
    lastKind: Object.create(null),
    mask: null, // {masked, found, counts, dossier, source, outdated}
    restore: null, // {text, unknown, restored}
    files: [],
    fileSeq: 0,
    filesBusy: 0,
    journal: [],
    journalLoaded: false,
  };

  /* ------------------------------------------------------------ small helpers */

  const $ = (sel, root) => (root || document).querySelector(sel);
  const enc = encodeURIComponent;
  const own = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

  function hasKey(key) {
    return own(I18N.fr, key);
  }

  /** Translate a key; {name} placeholders are filled from vars. */
  function t(key, vars) {
    const dict = I18N[state.lang] || I18N.fr;
    let s = own(dict, key) ? dict[key] : (hasKey(key) ? I18N.fr[key] : key);
    if (vars) {
      s = s.replace(/\{(\w+)\}/g, (m, name) => (own(vars, name) && vars[name] != null ? String(vars[name]) : ""));
    }
    if (state.lang === "fr") {
      // French typography: no line break before : ; ? ! » or after «
      s = s.replace(/ ([:;?!»])/g, "\u00a0$1").replace(/« /g, "«\u00a0");
    }
    return s;
  }

  function locale() {
    return state.lang === "fr" ? "fr-FR" : "en-GB";
  }

  function fmtNum(n) {
    const v = Number(n);
    return Number.isFinite(v) ? v.toLocaleString(locale()) : String(n);
  }

  /** Plural-aware translation: base_one / base_other. */
  function tn(base, n, vars) {
    const v = Number(n) || 0;
    const one = state.lang === "fr" ? Math.abs(v) < 2 : v === 1;
    return t(base + (one ? "_one" : "_other"), Object.assign({ n: fmtNum(v) }, vars || {}));
  }

  function kindLabel(kind, singular) {
    const key = (singular ? "kind1_" : "kind_") + kind;
    return hasKey(key) ? t(key) : String(kind || "");
  }

  function kindOfCode(code) {
    const s = String(code || "");
    const i = s.lastIndexOf("_");
    return i > 0 && own(PREFIX_KIND, s.slice(0, i)) ? PREFIX_KIND[s.slice(0, i)] : "";
  }

  function orderKinds(keys) {
    const known = KINDS.filter((k) => keys.includes(k));
    const other = keys.filter((k) => !KINDS.includes(k)).sort();
    return known.concat(other);
  }

  function countByKind(items) {
    const out = {};
    for (const it of items) {
      const k = it.kind || kindOfCode(it.code) || "?";
      out[k] = (out[k] || 0) + 1;
    }
    return out;
  }

  function summaryText(counts) {
    return orderKinds(Object.keys(counts))
      .filter((k) => counts[k] > 0)
      .map((k) => kindLabel(k) + " " + fmtNum(counts[k]))
      .join(", ");
  }

  /** Accepts {KIND: n} or {counts: {KIND: n}}; keeps numeric values only. */
  function normCounts(raw) {
    const out = {};
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return out;
    const src = raw.counts && typeof raw.counts === "object" && !Array.isArray(raw.counts) ? raw.counts : raw;
    for (const [k, v] of Object.entries(src)) {
      if (typeof v === "boolean") continue;
      const n = Number(v);
      if (v !== null && v !== "" && Number.isFinite(n)) out[k] = n;
    }
    return out;
  }

  function count(v) {
    if (Array.isArray(v)) return v.length;
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  }

  /** Lower case, no accents: for filters and the kept-clear check. */
  function fold(s) {
    return String(s == null ? "" : s).normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  }

  function baseName(p) {
    const parts = String(p || "").split(/[\\/]+/).filter(Boolean);
    return parts.length ? parts[parts.length - 1] : "";
  }

  function uniq(list) {
    return Array.from(new Set(list));
  }

  function dossierLabel(d) {
    const name = d.title || baseName(d.folder);
    return name ? d.code + " · " + name : String(d.code);
  }

  /** Clean a pasted path: surrounding quotes (Windows "Copy as path") and file:// URLs. */
  function cleanPath(raw) {
    let p = String(raw || "").trim().replace(/^["']+|["']+$/g, "").trim();
    if (/^file:\/\//i.test(p)) {
      try {
        p = decodeURIComponent(new URL(p).pathname);
        if (/^\/[A-Za-z]:[\\/]/.test(p)) p = p.slice(1);
      } catch (e) {
        /* keep the text as typed */
      }
    }
    return p;
  }

  function reduceMotion() {
    return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  function scrollToEl(el) {
    if (el && el.scrollIntoView) el.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" });
  }

  function cssEscape(s) {
    return window.CSS && window.CSS.escape ? window.CSS.escape(s) : String(s).replace(/["\\]/g, "\\$&");
  }

  /** Create an element. Text always goes in as text nodes, never as HTML. */
  function h(tag, props, ...children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = String(v);
        else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
        else if (v === true) node.setAttribute(k, "");
        else node.setAttribute(k, String(v));
      }
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  }

  /**
   * Rebuild a container, keep what the user was typing in its fields (data-fk), and give
   * focus and caret back to the same control if it had focus.
   */
  function preserveFocus(container, build) {
    const active = document.activeElement;
    const key = active && container.contains(active) ? active.getAttribute("data-fk") : null;
    let caret = null;
    if (key && typeof active.selectionStart === "number") caret = [active.selectionStart, active.selectionEnd];
    const typed = new Map();
    container.querySelectorAll("input[data-fk], select[data-fk]").forEach((el) => {
      if (el.type !== "checkbox") typed.set(el.getAttribute("data-fk"), el.value);
    });
    build();
    typed.forEach((value, fk) => {
      const el = container.querySelector('[data-fk="' + cssEscape(fk) + '"]');
      if (el && el.value !== value) el.value = value;
    });
    if (key) {
      const again = container.querySelector('[data-fk="' + cssEscape(key) + '"]');
      if (again) {
        again.focus({ preventScroll: true });
        if (caret && typeof again.setSelectionRange === "function") {
          try {
            again.setSelectionRange(caret[0], caret[1]);
          } catch (e) {
            /* not a text field */
          }
        }
      }
    }
  }

  /* ------------------------------------------------------------ API */

  class ApiError extends Error {
    constructor(status, message) {
      super(message || "");
      this.status = status; // -1: no answer, -2: unreadable answer, else HTTP status
    }
  }

  function serverMessage(data) {
    if (!data || typeof data !== "object") return "";
    let m = data.error || data.detail || data.message || "";
    if (Array.isArray(m)) m = m.map((x) => (x && x.msg) || String(x)).join(" ");
    if (typeof m !== "string") m = "";
    return m.length > 300 ? m.slice(0, 300) + "…" : m;
  }

  async function api(method, path, body) {
    const headers = { Accept: "application/json" };
    if (token) headers["X-Shield-Token"] = token;
    const init = { method, headers, cache: "no-store", credentials: "same-origin" };
    if (body instanceof FormData) {
      init.body = body; // the browser sets the multipart boundary
    } else if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    let res;
    try {
      res = await fetch(path, init);
    } catch (e) {
      setConn("off");
      throw new ApiError(-1, "");
    }
    let raw = "";
    try {
      raw = await res.text();
    } catch (e) {
      raw = "";
    }
    let data = null;
    let parsed = true;
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch (e) {
        parsed = false;
      }
    }
    if (!res.ok) {
      if (res.status === 401 || res.status === 403) setConn("token");
      throw new ApiError(res.status, parsed ? serverMessage(data) : "");
    }
    if (!parsed) throw new ApiError(-2, "");
    setConn("ok");
    return data;
  }

  function errorText(err) {
    if (!(err instanceof ApiError)) return t("err_generic");
    const s = err.status;
    if (s === -1) return t("err_offline");
    if (s === -2) return t("err_bad_response");
    if (s === 401 || s === 403) return t("err_token");
    if (s === 404) return t("err_notfound");
    if (s === 413) return t("err_too_big");
    if (s >= 400 && s < 500) return t("err_bad_request", { msg: err.message || String(s) });
    return t("err_server", { status: s, msg: err.message || "" }).trim();
  }

  /* ------------------------------------------------------------ alert, toast, connection */

  function renderAlert() {
    const box = $("#alert");
    const a = state.alert;
    if (!a) {
      box.hidden = true;
      $("#alert-text").textContent = "";
      return;
    }
    $("#alert-text").textContent = a.err ? errorText(a.err) : t(a.key);
    $("#alert-retry").hidden = !(a.err instanceof ApiError && a.err.status === -1);
    box.hidden = false;
  }

  function showError(err) {
    state.alert = { err };
    renderAlert();
  }

  function showAlertKey(key) {
    state.alert = { key };
    renderAlert();
  }

  function clearAlert() {
    state.alert = null;
    renderAlert();
  }

  let toastTimer = 0;
  function toast(message) {
    const el = $("#toast");
    el.textContent = "";
    window.clearTimeout(toastTimer);
    // Next frame, so a repeated message is announced again.
    window.requestAnimationFrame(() => {
      el.textContent = message;
      el.classList.add("show");
      toastTimer = window.setTimeout(() => el.classList.remove("show"), 4000);
    });
  }

  function setConn(mode) {
    if (state.conn === mode) return;
    state.conn = mode;
    renderConn();
    renderDossiers();
  }

  function renderConn() {
    const box = $("#conn");
    box.dataset.state = state.conn;
    let text;
    if (state.conn === "ok") text = state.version ? t("conn_ok_version", { v: state.version }) : t("conn_ok");
    else if (state.conn === "off") text = t("conn_off");
    else if (state.conn === "token") text = t("conn_token");
    else text = t("conn_checking");
    $("#conn-text").textContent = text;
  }

  /* ------------------------------------------------------------ busy buttons, clipboard, download */

  async function withBusy(btn, busyKey, fn) {
    if (!btn) return fn();
    if (btn.dataset.busy === "1") return undefined;
    const label = btn.textContent;
    btn.dataset.busy = "1";
    btn.setAttribute("aria-busy", "true");
    btn.classList.add("is-busy");
    if (busyKey) btn.textContent = t(busyKey);
    try {
      return await fn();
    } finally {
      delete btn.dataset.busy;
      btn.removeAttribute("aria-busy");
      btn.classList.remove("is-busy");
      if (busyKey) btn.textContent = btn.dataset.i18n ? t(btn.dataset.i18n) : label;
    }
  }

  function legacyCopy(text) {
    const active = document.activeElement;
    const ta = h("textarea", { class: "copy-buffer", readonly: true, "aria-hidden": "true", tabindex: "-1" });
    ta.value = text;
    document.body.append(ta);
    ta.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (e) {
      ok = false;
    }
    ta.remove();
    if (active && typeof active.focus === "function") active.focus({ preventScroll: true });
    return ok;
  }

  async function copyText(text, okKey) {
    let ok = false;
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        ok = true;
      } catch (e) {
        ok = false;
      }
    }
    if (!ok) ok = legacyCopy(text);
    if (ok) toast(t(okKey));
    else showAlertKey("copy_failed");
  }

  function txtName(maskedName) {
    let base = String(maskedName || "").replace(/[\\/:*?"<>|\u0000-\u001f]+/g, "_").trim();
    base = base.replace(/\.[A-Za-z0-9]{1,5}$/, "");
    return (base || t("files_default_name")) + ".txt";
  }

  function downloadText(name, text) {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = h("a", { href: url, download: name, class: "visually-hidden" });
    document.body.append(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  /** Show text with every Shield code highlighted (codes only, built as DOM nodes). */
  function renderCoded(el, text, unknown) {
    const frag = document.createDocumentFragment();
    let last = 0;
    for (const m of String(text).matchAll(CODE_RE)) {
      if (m.index > last) frag.append(text.slice(last, m.index));
      const bad = unknown && unknown.has(m[0]);
      frag.append(h("mark", { class: bad ? "code-mark is-unknown" : "code-mark", text: m[0] }));
      last = m.index + m[0].length;
    }
    if (last < text.length) frag.append(text.slice(last));
    el.replaceChildren(frag);
  }

  /* ------------------------------------------------------------ tabs */

  function selectTab(name, focus, skipLoad) {
    if (!TABS.includes(name)) name = "dossiers";
    state.tab = name;
    for (const tab of TABS) {
      const on = tab === name;
      const btn = document.getElementById("tab-" + tab);
      btn.setAttribute("aria-selected", String(on));
      btn.tabIndex = on ? 0 : -1;
      document.getElementById("panel-" + tab).hidden = !on;
    }
    const active = document.getElementById("tab-" + name);
    if (focus) {
      active.focus();
      active.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
    try {
      window.history.replaceState(window.history.state, "", "#" + name);
    } catch (e) {
      /* ignore */
    }
    if (name === "journal" && token && !skipLoad) loadJournal();
  }

  function bindTabs() {
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    tabs.forEach((tab, i) => {
      tab.addEventListener("click", () => selectTab(tab.dataset.tab));
      tab.addEventListener("keydown", (e) => {
        let j = null;
        if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
        else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = tabs.length - 1;
        if (j === null) return;
        e.preventDefault();
        selectTab(tabs[j].dataset.tab, true);
      });
    });
    window.addEventListener("hashchange", () => {
      const name = window.location.hash.slice(1);
      if (TABS.includes(name) && name !== state.tab) selectTab(name);
    });
  }

  /* ------------------------------------------------------------ language */

  function loadLang() {
    try {
      const v = window.localStorage.getItem(LANG_KEY);
      if (v === "fr" || v === "en") return v;
    } catch (e) {
      /* default */
    }
    return "fr";
  }

  function setLang(lang) {
    if (lang !== "fr" && lang !== "en") return;
    state.lang = lang;
    try {
      window.localStorage.setItem(LANG_KEY, lang);
    } catch (e) {
      /* not kept: fine */
    }
    applyStatic();
    renderAll();
  }

  function applyStatic() {
    document.documentElement.lang = state.lang;
    document.title = t("doc_title");
    document.querySelectorAll("[data-i18n]").forEach((n) => {
      if (n.dataset.busy !== "1") n.textContent = t(n.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((n) => n.setAttribute("placeholder", t(n.dataset.i18nPlaceholder)));
    document.querySelectorAll("[data-i18n-aria-label]").forEach((n) => n.setAttribute("aria-label", t(n.dataset.i18nAriaLabel)));
    document.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === state.lang)));
  }

  function renderAll() {
    renderConn();
    renderAlert();
    renderDossiers();
    renderSelects();
    renderMask();
    renderRestore();
    renderFiles();
    renderFilesBusy();
    renderJournal();
    renderExt();
  }

  /* ------------------------------------------------------------ browser extension panel */

  let tokenShown = false;

  /** The extension options need the Desk address and token: hidden by default, copy button. */
  function renderExt() {
    $("#ext-address").textContent = window.location.origin;
    const tok = $("#ext-token");
    tok.classList.toggle("is-missing", !token);
    tok.textContent = !token ? t("ext_no_token") : (tokenShown ? token : "\u2022".repeat(16));
    const show = $("#ext-show");
    show.hidden = !token;
    show.textContent = t(tokenShown ? "ext_hide" : "ext_show");
    show.setAttribute("aria-pressed", String(tokenShown));
    $("#ext-copy").hidden = !token;
  }

  /* ------------------------------------------------------------ dossiers */

  async function loadDossiers() {
    const list = await api("GET", "/api/dossiers");
    state.dossiers = (Array.isArray(list) ? list : [])
      .filter((d) => d && d.code)
      .map((d) => Object.assign({}, d, { code: String(d.code) }));
    const codes = new Set(state.dossiers.map((d) => d.code));
    if (state.current && !codes.has(state.current)) state.current = "";
    renderDossiers();
    renderSelects();
  }

  async function loadEntities(code) {
    const list = await api("GET", "/api/dossiers/" + enc(code) + "/entities");
    state.entities[code] = (Array.isArray(list) ? list : [])
      .filter((e) => e && e.code)
      .map((e) => ({
        code: String(e.code),
        kind: String(e.kind || kindOfCode(e.code) || ""),
        value: e.value == null ? "" : String(e.value),
        kept_clear: !!e.kept_clear,
      }));
    return state.entities[code];
  }

  function markMaskOutdated(code) {
    if (state.mask && state.mask.dossier === code) {
      state.mask.outdated = true;
      updateStale();
    }
  }

  function renderDossiers() {
    const box = $("#dossier-list");
    preserveFocus(box, () => {
      if (state.conn !== "ok") {
        // The list was not read (no token, Desk down): never say "no dossier yet", the user would add them again.
        const key = state.conn === "token" ? "dos_locked" : state.conn === "off" ? "dos_offline" : "dos_loading";
        box.replaceChildren(h("p", { class: "empty", text: t(key) }));
        return;
      }
      if (!state.dossiers.length) {
        box.replaceChildren(h("p", { class: "empty", text: t("dos_empty") }));
        return;
      }
      box.replaceChildren(...state.dossiers.map(dossierCard));
    });
  }

  function dossierCard(d) {
    const code = d.code;
    const open = !!state.open[code];
    const scanning = state.scanning.has(code);
    const detailId = "detail-" + code;
    const titleId = "dtitle-" + code;
    return h("article", { class: "card dossier", "data-code": code, "aria-labelledby": titleId },
      h("div", { class: "dossier-head" },
        h("div", { class: "dossier-id" },
          h("span", { class: "badge", text: code }),
          h("h3", { id: titleId, class: "dossier-title", tabindex: "-1", "data-fk": "title-" + code, text: d.title || baseName(d.folder) || code })),
        d.folder ? h("p", { class: "dossier-folder", text: String(d.folder) }) : null,
        h("p", { class: "dossier-meta", text: tn("dos_docs", count(d.documents)) + " · " + tn("dos_ents", count(d.entities)) })),
      state.scanMsg[code] ? h("p", { class: "scan-msg", text: state.scanMsg[code] }) : null,
      h("div", { class: "actions" },
        h("button", {
          type: "button", class: "btn", "aria-expanded": String(open), "aria-controls": detailId,
          "data-fk": "toggle-" + code, onclick: () => toggleDossier(code),
        }, t(open ? "dos_hide" : "dos_show")),
        h("button", {
          type: "button", class: scanning ? "btn is-busy" : "btn", "aria-busy": scanning ? "true" : null,
          "data-fk": "scan-" + code, onclick: () => scanDossier(code),
        }, t(scanning ? "dos_scanning" : "dos_scan")),
        h("button", {
          type: "button", class: "btn btn-danger", "data-fk": "remove-" + code, onclick: () => removeDossier(code),
        }, t("dos_remove"))),
      h("div", { id: detailId, class: "dossier-detail", hidden: !open }, open ? dossierDetail(code) : null));
  }

  function dossierDetail(code) {
    const ents = state.entities[code];
    if (!Array.isArray(ents)) return h("p", { class: "muted", text: t("loading") });
    const counts = countByKind(ents);
    const filterId = "filter-" + code;
    const groupsId = "groups-" + code;
    let filter = null;
    if (ents.length) {
      filter = h("input", Object.assign({
        type: "search", id: filterId, class: "filter", value: state.filter[code] || "",
        placeholder: t("ent_filter_ph"), "data-fk": "filter-" + code, "aria-controls": groupsId,
      }, PRIVATE));
      filter.addEventListener("input", () => {
        state.filter[code] = filter.value;
        renderGroups(code);
      });
    }
    return h("div", { class: "detail-inner" },
      ents.length
        ? h("ul", { class: "chips", "aria-label": t("dos_counts_label") },
          orderKinds(Object.keys(counts)).map((k) => h("li", { class: "chip" }, kindLabel(k) + " ", h("strong", { text: fmtNum(counts[k]) }))))
        : null,
      addNameForm("add-" + code, "add_name_title", "add_name_hint", "add_btn", async (kind, value) => {
        const ent = await api("POST", "/api/dossiers/" + enc(code) + "/entities", { kind, value });
        await loadEntities(code);
        const d = state.dossiers.find((x) => x.code === code);
        if (d && !Array.isArray(d.entities)) d.entities = state.entities[code].length;
        markMaskOutdated(code);
        renderDossiers();
        const input = document.getElementById("add-" + code + "-value");
        if (input) input.focus();
        toast(t("add_done", { code: ent && ent.code ? ent.code : "" }));
      }),
      filter ? h("div", { class: "field filter-field" }, h("label", { for: filterId, text: t("ent_filter_label") }), filter) : null,
      h("div", { id: groupsId, class: "groups" }, groupsContent(code)));
  }

  function renderGroups(code) {
    const box = document.getElementById("groups-" + code);
    if (box) box.replaceChildren(...groupsContent(code));
  }

  function groupsContent(code) {
    const ents = state.entities[code] || [];
    if (!ents.length) return [h("p", { class: "empty", text: t("ent_none") })];
    const q = fold(state.filter[code] || "").trim();
    const shown = q
      ? ents.filter((e) => fold(e.code).includes(q) || fold(e.value).includes(q) || fold(kindLabel(e.kind)).includes(q))
      : ents;
    if (!shown.length) return [h("p", { class: "empty", text: t("ent_no_match") })];
    const byKind = countByKind(shown);
    return orderKinds(Object.keys(byKind)).map((kind) => {
      const rows = shown.filter((e) => (e.kind || kindOfCode(e.code) || "?") === kind).sort(byCode);
      const gid = "g-" + code + "-" + kind;
      return h("section", { class: "kind-group", "aria-labelledby": gid },
        h("h4", { id: gid }, kindLabel(kind), " ", h("span", { class: "count", text: fmtNum(rows.length) })),
        h("div", { class: "table-wrap" },
          h("table", { class: "data" },
            h("thead", null, h("tr", null,
              h("th", { scope: "col", text: t("ent_col_code") }),
              h("th", { scope: "col", text: t("ent_col_value") }),
              h("th", { scope: "col", class: "col-toggle", text: t("ent_col_clear") }))),
            h("tbody", null, rows.map((e) => h("tr", { class: e.kept_clear ? "is-clear" : null },
              h("td", null, h("code", { class: "code", text: e.code })),
              h("td", { class: "value", text: e.value }),
              h("td", { class: "col-toggle" }, clearSwitch(code, e, "dos", () => markMaskOutdated(code)))))))));
    });
  }

  function byCode(a, b) {
    return String(a.code).localeCompare(String(b.code), "en", { numeric: true });
  }

  /** "Garder en clair" switch for one entity. */
  function clearSwitch(dcode, ent, scope, after) {
    const input = h("input", {
      type: "checkbox", class: "switch", checked: !!ent.kept_clear,
      "aria-label": t("ent_clear_aria", { code: ent.code }), "data-fk": scope + "-clear-" + ent.code,
    });
    input.addEventListener("click", (e) => {
      if (input.dataset.busy === "1") e.preventDefault();
    });
    input.addEventListener("change", async () => {
      const want = input.checked;
      input.dataset.busy = "1";
      input.setAttribute("aria-busy", "true");
      clearAlert();
      try {
        await api("POST", "/api/dossiers/" + enc(dcode) + "/keep-clear", { code: ent.code, value: want });
        ent.kept_clear = want;
        const row = input.closest("tr");
        if (row) row.classList.toggle("is-clear", want);
        toast(t(want ? "ent_kept_on" : "ent_kept_off", { code: ent.code }));
        if (after) await after(want);
      } catch (err) {
        input.checked = !want;
        showError(err);
      } finally {
        delete input.dataset.busy;
        input.removeAttribute("aria-busy");
      }
    });
    return input;
  }

  /** Form to register a name or term in a dossier. */
  function addNameForm(prefix, legendKey, hintKey, btnKey, onAdd) {
    const kindId = prefix + "-kind";
    const valueId = prefix + "-value";
    const hintId = prefix + "-hint";
    const kind = h("select", { id: kindId, "data-fk": kindId },
      KINDS.map((k) => h("option", { value: k }, kindLabel(k, true))));
    kind.value = state.lastKind[prefix] || "PERSON";
    kind.addEventListener("change", () => {
      state.lastKind[prefix] = kind.value;
    });
    const value = h("input", Object.assign({
      type: "text", id: valueId, required: true, placeholder: t("add_value_ph"), "data-fk": valueId,
      "aria-describedby": hintKey ? hintId : null,
    }, PRIVATE));
    const btn = h("button", { type: "submit", class: "btn", "data-fk": prefix + "-btn" }, t(btnKey));
    const form = h("form", { class: "add-name", novalidate: true },
      h("fieldset", null,
        h("legend", { text: t(legendKey) }),
        hintKey ? h("p", { id: hintId, class: "field-hint", text: t(hintKey) }) : null,
        h("div", { class: "add-row" },
          h("div", { class: "field" }, h("label", { for: kindId, text: t("add_kind_label") }), kind),
          h("div", { class: "field" }, h("label", { for: valueId, text: t("add_value_label") }), value),
          h("div", { class: "field field-btn" }, btn))));
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const v = value.value.trim();
      if (!v) {
        value.setAttribute("aria-invalid", "true");
        value.focus();
        toast(t("add_empty"));
        return;
      }
      value.removeAttribute("aria-invalid");
      clearAlert();
      // Clear first: the add re-renders this form and would otherwise carry the value over.
      value.value = "";
      await withBusy(btn, null, async () => {
        try {
          await onAdd(kind.value, v);
        } catch (err) {
          const field = document.getElementById(valueId) || value;
          if (!field.value) field.value = v;
          showError(err);
        }
      });
    });
    return form;
  }

  async function toggleDossier(code) {
    state.open[code] = !state.open[code];
    renderDossiers();
    if (state.open[code] && !Array.isArray(state.entities[code])) {
      try {
        await loadEntities(code);
      } catch (err) {
        state.open[code] = false;
        showError(err);
      }
      renderDossiers();
    }
  }

  function scanSummary(res, ents) {
    const raw = normCounts(res);
    const kinds = {};
    for (const k of Object.keys(raw)) if (KINDS.includes(k) && raw[k] > 0) kinds[k] = raw[k];
    let summary = summaryText(kinds);
    if (!summary && Array.isArray(ents) && ents.length) summary = summaryText(countByKind(ents));
    return summary ? t("dos_scan_done", { summary }) : t("dos_scan_none");
  }

  async function scanDossier(code) {
    if (state.scanning.has(code)) return;
    clearAlert();
    state.scanning.add(code);
    renderDossiers();
    try {
      const res = await api("POST", "/api/dossiers/" + enc(code) + "/scan");
      try {
        await loadEntities(code);
      } catch (e) {
        /* the summary then comes from the scan answer only */
      }
      state.open[code] = true;
      state.scanMsg[code] = scanSummary(res, state.entities[code]);
      markMaskOutdated(code);
      state.scanning.delete(code);
      await loadDossiers();
      toast(state.scanMsg[code]);
    } catch (err) {
      showError(err);
    } finally {
      state.scanning.delete(code);
      renderDossiers();
    }
  }

  async function removeDossier(code) {
    if (!window.confirm(t("dos_remove_confirm", { code }))) return;
    clearAlert();
    try {
      await api("DELETE", "/api/dossiers/" + enc(code));
      for (const map of [state.entities, state.open, state.scanMsg, state.filter]) delete map[code];
      if (state.current === code) state.current = "";
      if (state.mask && state.mask.dossier === code) {
        state.mask = null;
        renderMask();
      }
      await loadDossiers();
      toast(t("dos_removed", { code }));
      $("#dos-list-title").focus();
    } catch (err) {
      showError(err);
    }
  }

  async function onAddDossier(ev) {
    ev.preventDefault();
    const folderEl = $("#add-folder");
    const titleEl = $("#add-title");
    const folder = cleanPath(folderEl.value);
    const title = titleEl.value.trim();
    if (!folder) {
      folderEl.setAttribute("aria-invalid", "true");
      folderEl.focus();
      toast(t("dos_folder_empty"));
      return;
    }
    folderEl.removeAttribute("aria-invalid");
    clearAlert();
    await withBusy($("#add-btn"), "dos_adding", async () => {
      let created;
      try {
        created = await api("POST", "/api/dossiers", title ? { folder, title } : { folder });
      } catch (err) {
        showError(err);
        return;
      }
      folderEl.value = "";
      titleEl.value = "";
      const code = created && created.code ? String(created.code) : "";
      if (code) state.current = code;
      try {
        await loadDossiers();
      } catch (err) {
        showError(err);
        return;
      }
      toast(t("dos_added", { code }));
      if (!code) return;
      const heading = document.getElementById("dtitle-" + code);
      if (heading) {
        scrollToEl(heading.closest("article"));
        heading.focus({ preventScroll: true });
      }
      await scanDossier(code);
    });
  }

  /* ------------------------------------------------------------ dossier selectors */

  function renderSelects() {
    fillDossierSelect($("#mask-dossier"), "dossier_none");
    fillDossierSelect($("#files-dossier"), "dossier_none");
    fillDossierSelect($("#restore-dossier"), "dossier_pick");
    const j = $("#journal-dossier");
    const keep = j.value;
    j.replaceChildren(h("option", { value: "" }, t("journal_all")),
      ...state.dossiers.map((d) => h("option", { value: d.code }, dossierLabel(d))));
    j.value = state.dossiers.some((d) => d.code === keep) ? keep : "";
    renderHints();
  }

  function fillDossierSelect(sel, emptyKey) {
    sel.replaceChildren(h("option", { value: "" }, t(emptyKey)),
      ...state.dossiers.map((d) => h("option", { value: d.code }, dossierLabel(d))));
    syncSelect(sel);
  }

  function lastDossier() {
    try {
      const v = window.localStorage.getItem(DOSSIER_KEY) || "";
      return /^DOSSIER_[0-9]{1,6}$/.test(v) ? v : "";
    } catch (e) {
      return "";
    }
  }

  function rememberDossier(code) {
    try {
      if (code) window.localStorage.setItem(DOSSIER_KEY, code);
    } catch (e) {
      /* storage blocked: the choice lasts for this page only */
    }
  }

  /** Masquer, Fichiers and Restaurer start on the dossier used last, or on the only one: masking without a
   *  dossier by mistake would make the answer impossible to restore. */
  function syncSelect(sel) {
    const has = (code) => code && state.dossiers.some((d) => d.code === code);
    if (!state.current && !state.currentSet) {
      const last = lastDossier();
      if (has(last)) state.current = last;
      else if (state.dossiers.length === 1) state.current = state.dossiers[0].code;
    }
    sel.value = has(state.current) ? state.current : "";
  }

  function onDossierSelect(ev) {
    const sel = ev.target;
    state.current = sel.value;
    state.currentSet = true; // the user chose (possibly "no dossier"): keep that choice
    rememberDossier(sel.value);
    for (const id of ["mask-dossier", "files-dossier", "restore-dossier"]) {
      const other = document.getElementById(id);
      if (other !== sel) syncSelect(other);
    }
    if (sel.id === "mask-dossier" && state.mask && state.mask.dossier !== sel.value) {
      state.mask = null; // the result belonged to the other dossier
      renderMask();
    } else {
      updateStale();
    }
    renderHints();
  }

  function renderHints() {
    const maskD = $("#mask-dossier").value;
    const mh = $("#mask-dossier-hint");
    mh.textContent = t(maskD ? "mask_hint_dossier" : "mask_hint_none");
    mh.classList.toggle("is-warn", !maskD);
    const fh = $("#files-dossier-hint");
    const filesD = $("#files-dossier").value;
    fh.textContent = t(filesD ? "files_hint_dossier" : "files_hint_none");
    fh.classList.toggle("is-warn", !filesD);
    $("#restore-dossier-hint").textContent = t(state.dossiers.length ? "restore_hint_pick" : "restore_hint_none");
  }

  /* ------------------------------------------------------------ masquer */

  function normFound(f) {
    if (typeof f === "string") return { code: f, kind: kindOfCode(f), value: "" };
    if (!f || typeof f !== "object" || !f.code) return null;
    const code = String(f.code);
    return { code, kind: String(f.kind || kindOfCode(code) || ""), value: f.value == null ? "" : String(f.value) };
  }

  function reviewCounts(m) {
    const c = {};
    for (const [k, v] of Object.entries(m.counts)) if (v > 0) c[k] = v;
    return Object.keys(c).length ? c : countByKind(m.found);
  }

  function totalOf(counts) {
    return Object.values(counts).reduce((a, b) => a + b, 0);
  }

  async function doMask(opts) {
    const input = $("#mask-input");
    const text = input.value;
    if (!text.trim()) {
      toast(t("mask_empty"));
      input.focus();
      return;
    }
    const dossier = $("#mask-dossier").value;
    clearAlert();
    await withBusy($("#mask-btn"), "masking", async () => {
      try {
        const res = await api("POST", "/api/mask", dossier ? { text, dossier } : { text });
        // Fail closed: without a masked text in the answer, show nothing.
        if (!res || typeof res.masked !== "string") throw new ApiError(-2, "");
        const found = (Array.isArray(res.found) ? res.found : []).map(normFound).filter(Boolean);
        if (dossier) {
          try {
            await loadEntities(dossier);
          } catch (e) {
            /* the review then shows codes only */
          }
        }
        state.mask = { masked: res.masked, found, counts: normCounts(res.counts), dossier, source: text, outdated: false };
        renderMask();
        if (dossier) renderDossiers();
        toast(tn("mask_done", totalOf(reviewCounts(state.mask))));
        if (!(opts && opts.quiet)) scrollToEl($("#mask-result"));
      } catch (err) {
        showError(err);
      }
    });
  }

  function updateStale() {
    const m = state.mask;
    $("#mask-stale").hidden = !m || (!m.outdated && $("#mask-input").value === m.source
      && $("#mask-dossier").value === m.dossier);
  }

  function renderMask() {
    const card = $("#mask-result");
    const m = state.mask;
    if (!m) {
      card.hidden = true;
      $("#mask-output").replaceChildren();
      $("#mask-review").replaceChildren();
      return;
    }
    card.hidden = false;
    renderCoded($("#mask-output"), m.masked, null);
    updateStale();
    preserveFocus($("#mask-review"), renderReview);
  }

  function renderReview() {
    const m = state.mask;
    const nodes = [];
    const counts = reviewCounts(m);
    const total = totalOf(counts);
    nodes.push(h("p", { class: "review-summary" },
      total ? tn("review_summary", total, { summary: summaryText(counts) }) : t("review_none")));

    const ents = m.dossier && Array.isArray(state.entities[m.dossier]) ? state.entities[m.dossier] : [];
    const byCodeMap = new Map(ents.map((e) => [e.code, e]));
    const rows = [];
    const seen = new Set();
    for (const f of m.found) {
      if (seen.has(f.code)) continue;
      seen.add(f.code);
      const e = byCodeMap.get(f.code) || null;
      rows.push({ code: f.code, kind: f.kind || (e && e.kind) || "", value: e ? e.value : f.value, ent: e });
    }
    if (m.dossier) {
      // Terms kept in clear that appear in the text: listed so the user can mask them again.
      const src = fold(m.source);
      for (const e of ents) {
        if (!e.kept_clear || seen.has(e.code) || !e.value) continue;
        if (src.includes(fold(e.value))) {
          seen.add(e.code);
          rows.push({ code: e.code, kind: e.kind, value: e.value, ent: e });
        }
      }
    }
    rows.sort((a, b) => {
      const ka = KINDS.indexOf(a.kind);
      const kb = KINDS.indexOf(b.kind);
      return (ka < 0 ? 99 : ka) - (kb < 0 ? 99 : kb) || byCode(a, b);
    });
    if (rows.length) nodes.push(h("div", { class: "table-wrap" }, reviewTable(m, rows)));

    if (!m.dossier) {
      nodes.push(h("p", { class: "note", text: t("review_nodossier") }));
    } else {
      const dcode = m.dossier;
      nodes.push(addNameForm("mask-add", "review_add_title", null, "review_add_btn", async (kind, value) => {
        const ent = await api("POST", "/api/dossiers/" + enc(dcode) + "/entities", { kind, value });
        try {
          await loadEntities(dcode);
        } catch (e) {
          /* doMask reloads them too */
        }
        renderDossiers();
        await doMask({ quiet: true });
        const again = document.getElementById("mask-add-value");
        if (again) again.focus();
        toast(t("add_done", { code: ent && ent.code ? ent.code : "" }));
      }));
    }
    $("#mask-review").replaceChildren(...nodes);
  }

  function reviewTable(m, rows) {
    const withValues = rows.some((r) => r.value);
    const withToggle = !!m.dossier;
    // With values or switches, the type goes under the code so the table fits a phone screen.
    const compact = withValues || withToggle;
    return h("table", { class: "data", "aria-label": t("review_table_label") },
      h("thead", null, h("tr", null,
        h("th", { scope: "col", text: compact ? t("ent_col_code") + " · " + t("ent_col_kind") : t("ent_col_code") }),
        compact ? null : h("th", { scope: "col", text: t("ent_col_kind") }),
        withValues ? h("th", { scope: "col", text: t("ent_col_value") }) : null,
        withToggle ? h("th", { scope: "col", class: "col-toggle", text: t("ent_col_clear") }) : null)),
      h("tbody", null, rows.map((r) => {
        const clear = !!(r.ent && r.ent.kept_clear);
        const kind = r.kind ? kindLabel(r.kind, true) : "";
        return h("tr", { class: clear ? "is-clear" : null },
          compact
            ? h("td", null, h("code", { class: "code", text: r.code }), kind ? h("span", { class: "sub", text: kind }) : null)
            : h("td", null, h("code", { class: "code", text: r.code })),
          compact ? null : h("td", { text: kind }),
          withValues ? h("td", { class: "value" }, r.value || "", clear ? h("span", { class: "tag", text: t("review_kept") }) : null) : null,
          withToggle
            ? h("td", { class: "col-toggle" }, r.ent ? clearSwitch(m.dossier, r.ent, "mask", async () => {
              renderDossiers();
              await doMask({ quiet: true });
            }) : null)
            : null);
      })));
  }

  /* ------------------------------------------------------------ restaurer */

  async function doRestore() {
    const input = $("#restore-input");
    const sel = $("#restore-dossier");
    const text = input.value;
    const dossier = sel.value;
    if (!dossier) {
      toast(t(state.dossiers.length ? "restore_need_dossier" : "restore_hint_none"));
      sel.focus();
      return;
    }
    if (!text.trim()) {
      toast(t("restore_empty"));
      input.focus();
      return;
    }
    clearAlert();
    await withBusy($("#restore-btn"), "restoring", async () => {
      try {
        const res = await api("POST", "/api/restore", { text, dossier });
        if (!res || typeof res.text !== "string") throw new ApiError(-2, "");
        const unknown = uniq((Array.isArray(res.unknown_codes) ? res.unknown_codes : []).map(String));
        const n = Number(res.restored);
        state.restore = { text: res.text, unknown, restored: Number.isFinite(n) ? n : 0 };
        renderRestore();
        toast(tn("restore_count", state.restore.restored));
        scrollToEl($("#restore-result"));
      } catch (err) {
        showError(err);
      }
    });
  }

  function renderRestore() {
    const card = $("#restore-result");
    const r = state.restore;
    if (!r) {
      card.hidden = true;
      $("#restore-output").replaceChildren();
      return;
    }
    card.hidden = false;
    $("#restore-summary").textContent = tn("restore_count", r.restored);
    $("#restore-unknown").hidden = !r.unknown.length;
    $("#restore-unknown-list").replaceChildren(...r.unknown.map((c) => {
      const kind = kindOfCode(c);
      return h("li", null, h("code", { class: "code", text: c }), kind ? " (" + kindLabel(kind, true) + ")" : null);
    }));
    renderCoded($("#restore-output"), r.text, new Set(r.unknown));
  }

  /* ------------------------------------------------------------ fichiers */

  async function uploadFiles(list) {
    const files = Array.from(list || []);
    if (!files.length || state.filesBusy) return;
    clearAlert();
    const fd = new FormData();
    for (const f of files) fd.append("files", f, f.name);
    const dossier = $("#files-dossier").value;
    if (dossier) fd.append("dossier", dossier);
    state.filesBusy = files.length;
    renderFilesBusy();
    try {
      const res = await api("POST", "/api/files", fd);
      const items = Array.isArray(res) ? res : (res && Array.isArray(res.files) ? res.files : null);
      if (!items) throw new ApiError(-2, "");
      const added = items.filter((x) => x && typeof x === "object").map((x) => ({
        id: ++state.fileSeq,
        name: x.name == null ? "" : String(x.name),
        masked_name: x.masked_name == null ? "" : String(x.masked_name),
        masked_text: typeof x.masked_text === "string" ? x.masked_text : "",
        error: x.error ? String(x.error) : "",
      }));
      state.files = added.concat(state.files);
      renderFiles();
      toast(tn("files_done", added.filter((f) => !f.error && f.masked_text).length));
    } catch (err) {
      showError(err);
    } finally {
      state.filesBusy = 0;
      renderFilesBusy();
      $("#file-input").value = "";
    }
  }

  function renderFilesBusy() {
    $("#files-status").textContent = state.filesBusy ? tn("files_sending", state.filesBusy) : "";
    const dz = $("#dropzone");
    dz.classList.toggle("is-busy", !!state.filesBusy);
    if (state.filesBusy) dz.setAttribute("aria-busy", "true");
    else dz.removeAttribute("aria-busy");
  }

  function renderFiles() {
    $("#files-results").hidden = !state.files.length;
    const list = $("#files-list");
    preserveFocus(list, () => list.replaceChildren(...state.files.map(fileItem)));
  }

  function fileItem(f) {
    const text = f.masked_text;
    let status = null;
    if (f.error) status = h("p", { class: "note note-error", text: f.error });
    else if (!text) status = h("p", { class: "note note-warn", text: t("files_no_text") });
    let preview = null;
    if (text) {
      const out = h("div", { class: "output output-small" });
      preview = h("details", { class: "preview" },
        h("summary", { "data-fk": "file-preview-" + f.id, text: t("files_preview") }),
        text.length > PREVIEW_CHARS ? h("p", { class: "field-hint", text: t("files_preview_cut", { n: fmtNum(PREVIEW_CHARS) }) }) : null,
        out);
      // Build the preview only when it is opened (large documents).
      preview.addEventListener("toggle", () => {
        if (preview.open && !out.childNodes.length) renderCoded(out, text.length > PREVIEW_CHARS ? text.slice(0, PREVIEW_CHARS) : text, null);
      });
    }
    return h("li", { class: "file-item" },
      h("dl", { class: "file-names" },
        h("div", null, h("dt", { text: t("files_original") }), h("dd", { text: f.name || "?" })),
        h("div", null, h("dt", { text: t("files_masked_name") }), h("dd", { class: "mono", text: f.masked_name || txtName("") }))),
      status,
      text
        ? h("div", { class: "actions" },
          h("button", {
            type: "button", class: "btn btn-primary", "data-fk": "file-dl-" + f.id,
            onclick: () => downloadText(txtName(f.masked_name), text),
          }, t("files_download")),
          h("button", {
            type: "button", class: "btn", "data-fk": "file-copy-" + f.id, onclick: () => copyText(text, "files_copied"),
          }, t("files_copy")))
        : null,
      preview);
  }

  function hasFiles(e) {
    const types = e.dataTransfer && e.dataTransfer.types;
    return !!types && Array.from(types).includes("Files");
  }

  function bindFiles() {
    const input = $("#file-input");
    $("#files-pick").addEventListener("click", () => input.click());
    input.addEventListener("change", () => uploadFiles(input.files));
    const panel = $("#panel-fichiers");
    const dz = $("#dropzone");
    let depth = 0;
    panel.addEventListener("dragenter", (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth += 1;
      dz.classList.add("is-over");
    });
    panel.addEventListener("dragover", (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    });
    panel.addEventListener("dragleave", () => {
      depth = Math.max(0, depth - 1);
      if (!depth) dz.classList.remove("is-over");
    });
    panel.addEventListener("drop", (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      depth = 0;
      dz.classList.remove("is-over");
      uploadFiles(e.dataTransfer.files);
    });
    // A file dropped anywhere else must not replace this page.
    window.addEventListener("dragover", (e) => {
      if (hasFiles(e)) e.preventDefault();
    });
    window.addEventListener("drop", (e) => {
      if (hasFiles(e)) e.preventDefault();
    });
    $("#files-clear").addEventListener("click", () => {
      state.files = [];
      renderFiles();
      $("#files-pick").focus();
    });
  }

  /* ------------------------------------------------------------ journal */

  async function loadJournal() {
    const d = $("#journal-dossier").value;
    await withBusy($("#journal-refresh"), null, async () => {
      try {
        const rows = await api("GET", "/api/journal" + (d ? "?dossier=" + enc(d) : ""));
        state.journal = Array.isArray(rows) ? rows : (rows && Array.isArray(rows.events) ? rows.events : []);
        state.journalLoaded = true;
        renderJournal();
      } catch (err) {
        showError(err);
      }
    });
  }

  function toMs(v) {
    if (typeof v === "number") return v < 1e12 ? v * 1000 : v;
    if (typeof v === "string" && v) {
      if (/^\d+(\.\d+)?$/.test(v)) return toMs(Number(v));
      const ms = Date.parse(v);
      return Number.isNaN(ms) ? NaN : ms;
    }
    return NaN;
  }

  function fmtTime(v) {
    const ms = toMs(v);
    if (Number.isNaN(ms)) return v == null ? "" : String(v);
    try {
      return new Date(ms).toLocaleString(locale(), { dateStyle: "short", timeStyle: "medium" });
    } catch (e) {
      return new Date(ms).toLocaleString();
    }
  }

  function eventLabel(ev) {
    const key = "ev_" + String(ev || "").toLowerCase().replace(/[^a-z0-9]+/g, "_");
    return hasKey(key) ? t(key) : String(ev == null ? "" : ev);
  }

  function countsText(c) {
    if (c == null) return "";
    if (typeof c !== "object") return typeof c === "number" ? fmtNum(c) : String(c);
    const parts = [];
    for (const k of orderKinds(Object.keys(c))) {
      const v = c[k];
      if (v === null || v === undefined || typeof v === "object") continue;
      const label = KINDS.includes(k) ? kindLabel(k) : (hasKey("count_" + k) ? t("count_" + k) : k);
      parts.push(label + " " + (typeof v === "number" ? fmtNum(v) : String(v)));
    }
    return parts.join(" · ");
  }

  /** "dossier_added" (registry) and "add_dossier" (scan) describe one action: show one row. */
  function dedupeJournal(rows) {
    const adds = rows.filter((r) => r.event === "add_dossier").map((r) => [r.dossier, toMs(r.time)]);
    return rows.filter((r) => r.event !== "dossier_added"
      || !adds.some(([d, ms]) => d === r.dossier && Math.abs(ms - toMs(r.time)) < 10000));
  }

  function renderJournal() {
    const rows = dedupeJournal(state.journal.filter((r) => r && typeof r === "object"))
      .map((r, i) => ({ r, i, ms: toMs(r.time) }))
      .sort((a, b) => {
        if (Number.isNaN(a.ms) || Number.isNaN(b.ms)) return a.i - b.i;
        return b.ms - a.ms || a.i - b.i;
      })
      .map((x) => x.r);
    $("#journal-body").replaceChildren(...rows.map((r) => h("tr", null,
      h("td", { class: "nowrap", text: fmtTime(r.time) }),
      h("td", { text: eventLabel(r.event) }),
      h("td", null, r.dossier ? h("code", { class: "code", text: String(r.dossier) }) : null),
      h("td", { text: countsText(r.counts) }))));
    $("#journal-wrap").hidden = !rows.length;
    $("#journal-empty").hidden = !state.journalLoaded || rows.length > 0;
  }

  /* ------------------------------------------------------------ wiring */

  function onCtrlEnter(el, fn) {
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        fn();
      }
    });
  }

  function bindStatic() {
    document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
    $("#alert-close").addEventListener("click", clearAlert);
    $("#alert-retry").addEventListener("click", connect);
    bindTabs();

    $("#add-dossier-form").addEventListener("submit", onAddDossier);
    $("#dos-refresh").addEventListener("click", (e) => withBusy(e.currentTarget, null, async () => {
      clearAlert();
      try {
        await loadDossiers();
        await Promise.all(Object.keys(state.open).filter((c) => state.open[c]).map((c) => loadEntities(c)));
        renderDossiers();
      } catch (err) {
        showError(err);
      }
    }));

    for (const id of ["mask-dossier", "files-dossier", "restore-dossier"]) {
      document.getElementById(id).addEventListener("change", onDossierSelect);
    }

    const maskInput = $("#mask-input");
    $("#mask-btn").addEventListener("click", () => doMask());
    onCtrlEnter(maskInput, () => doMask());
    maskInput.addEventListener("input", updateStale);
    $("#mask-clear").addEventListener("click", () => {
      maskInput.value = "";
      state.mask = null;
      renderMask();
      maskInput.focus();
    });
    $("#mask-copy").addEventListener("click", () => {
      if (state.mask) copyText(state.mask.masked, "mask_copied");
    });

    const restoreInput = $("#restore-input");
    $("#restore-btn").addEventListener("click", () => doRestore());
    onCtrlEnter(restoreInput, () => doRestore());
    $("#restore-clear").addEventListener("click", () => {
      restoreInput.value = "";
      state.restore = null;
      renderRestore();
      restoreInput.focus();
    });
    $("#restore-copy").addEventListener("click", () => {
      if (state.restore) copyText(state.restore.text, "restore_copied");
    });

    bindFiles();

    $("#ext-copy").addEventListener("click", () => {
      if (token) copyText(token, "ext_copied");
    });
    $("#ext-show").addEventListener("click", () => {
      tokenShown = !tokenShown;
      renderExt();
    });

    $("#journal-dossier").addEventListener("change", () => loadJournal());
    $("#journal-refresh").addEventListener("click", () => loadJournal());
  }

  async function connect() {
    clearAlert();
    if (!token) {
      setConn("token");
      showAlertKey("err_token");
      return;
    }
    setConn("checking");
    try {
      const st = await api("GET", "/api/status");
      state.version = st && typeof st === "object" && st.version ? String(st.version) : "";
      renderConn();
      await loadDossiers();
      if (state.tab === "journal") await loadJournal();
    } catch (err) {
      showError(err);
    }
  }

  function boot() {
    state.lang = loadLang();
    bindStatic();
    applyStatic();
    renderAll();
    const initial = window.location.hash.slice(1);
    selectTab(TABS.includes(initial) ? initial : "dossiers", false, true);
    connect(); // also loads the journal when that tab is open
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
