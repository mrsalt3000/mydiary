/*
 * Local Diary — chat page behaviour (Tasks 06/07/09).
 *
 * Scope: the text compose form (Task 06), voice messages (Task 07):
 * recording through MediaRecorder when available, a file-upload fallback
 * through a hidden <input type=file>, upload to POST /api/v1/audio and
 * insertion of the created voice message into the chat list; the manual
 * transcription flow (Task 09): POST .../transcribe on button click, job
 * polling and raw-text rendering below the audio player; and deletion of a
 * message (task T15, ADR-011): an inline confirmation step, then
 * DELETE /api/v1/messages/{id} and removal of the message from the list. The
 * "load older messages" paging stays declarative (htmx server-rendered
 * fragments).
 *
 * Safety rules (must be preserved by any later edit):
 *  - Diary text and audio metadata are NEVER placed into the DOM as markup.
 *    New messages are built with createElement + textContent/properties only
 *    — no HTML-string sinks anywhere in this file.
 *  - The audio element <src> always points at the server-generated playback
 *    URL "/api/v1/audio/<uuid>/file" derived from the 201 response; no other
 *    value is ever accepted.
 *  - Error/info strings shown to the user are static; server-provided error
 *    payloads are never echoed into the page.
 *  - No fetch/script/link to external hosts: the page only talks to this app.
 *  - Deletion is irreversible (the message text and, for a voice message, the
 *    audio file are erased), so it only ever runs behind the explicit
 *    confirmation step; a 404 means "already deleted" and is not shown as a
 *    failure.
 */
"use strict";

(function () {
  var form = document.getElementById("compose-form");
  var textarea = document.getElementById("message-text");
  var errorBox = document.getElementById("form-error");
  var submitButton = document.getElementById("compose-submit");
  var messageList = document.getElementById("message-list");
  var emptyState = document.getElementById("message-empty");

  if (!form || !textarea || !errorBox || !submitButton || !messageList) {
    return;
  }

  var maxLength = Number(form.dataset.maxLength || 0);

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.hidden = true;
  }

  /* Server timestamps are ISO-8601 with the application offset already
     applied (e.g. "2026-09-10T05:45:00+03:00"). Render the wall-clock part
     in the same "%d.%m.%Y %H:%M" shape the server uses — the offset is read
     from the string itself, so the browser's own timezone never matters. */
  function formatTimestamp(iso) {
    var match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso);
    if (!match) {
      return iso;
    }
    return match[3] + "." + match[2] + "." + match[1] + " " + match[4] + ":" + match[5];
  }

  function messageFooter(message) {
    var meta = document.createElement("header");
    meta.className = "message__meta";
    var time = document.createElement("time");
    time.dateTime = message.created_at;
    time.textContent = formatTimestamp(message.created_at);
    meta.appendChild(time);
    return meta;
  }

  /* Delete controls of one message (task T15, ADR-011): mirrors
     ``_message_actions.html`` so a freshly appended message behaves exactly
     like a server-rendered one. Built with createElement/textContent only —
     no HTML-string sinks. */
  function buildMessageActions(messageId, createdAtIso) {
    var box = document.createElement("div");
    box.className = "message__actions";

    var open = document.createElement("button");
    open.type = "button";
    open.className = "message-action message-action--delete";
    open.setAttribute("data-action", "delete-message");
    open.setAttribute("data-message-id", messageId);
    open.setAttribute("aria-label", "Удалить запись от " + formatTimestamp(createdAtIso));
    open.textContent = "Удалить";

    var panel = document.createElement("div");
    panel.className = "message__confirm";
    panel.setAttribute("data-role", "delete-confirm");
    panel.setAttribute("role", "group");
    panel.setAttribute("aria-label", "Подтверждение удаления записи");
    panel.hidden = true;

    var text = document.createElement("p");
    text.className = "message__confirm-text";
    text.textContent =
      "Удалить запись? Текст и голосовой файл будут стёрты без возможности восстановления.";

    var confirm = document.createElement("button");
    confirm.type = "button";
    confirm.className = "message-action message-action--danger";
    confirm.setAttribute("data-action", "delete-message-confirm");
    confirm.textContent = "Да, удалить";

    var cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "message-action";
    cancel.setAttribute("data-action", "delete-message-cancel");
    cancel.textContent = "Отмена";

    var status = document.createElement("p");
    status.className = "message__note";
    status.setAttribute("data-role", "delete-status");
    status.setAttribute("role", "status");
    status.hidden = true;

    panel.appendChild(text);
    panel.appendChild(confirm);
    panel.appendChild(cancel);
    box.appendChild(open);
    box.appendChild(panel);
    box.appendChild(status);
    return box;
  }

  function appendMessage(message) {
    if (emptyState && emptyState.parentNode === messageList) {
      emptyState.remove();
    }

    var article = document.createElement("article");
    article.className = "message";
    article.id = "message-" + message.id;

    var meta = messageFooter(message);
    var body = document.createElement("div");
    body.className = "message__body";
    body.textContent = message.body; // diary text: textContent only, never markup

    article.appendChild(meta);
    article.appendChild(body);
    article.appendChild(buildMessageActions(message.id, message.created_at));
    messageList.appendChild(article);
    article.scrollIntoView({ block: "nearest" });
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    var body = textarea.value;

    if (!body.trim()) {
      showError("Запись не может быть пустой.");
      textarea.focus();
      return;
    }
    if (maxLength > 0 && body.length > maxLength) {
      showError("Запись слишком длинная — не более " + maxLength + " символов.");
      textarea.focus();
      return;
    }

    clearError();
    submitButton.disabled = true;
    form.setAttribute("aria-busy", "true");

    fetch("/api/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body: body }),
    })
      .then(function (response) {
        if (response.status === 201) {
          return response.json().then(function (created) {
            textarea.value = "";
            appendMessage(created);
          });
        }
        throw new Error("create-failed");
      })
      .catch(function () {
        // Static, friendly message. Server error details are never rendered.
        showError("Не удалось сохранить запись. Попробуйте ещё раз.");
      })
      .finally(function () {
        submitButton.disabled = false;
        form.removeAttribute("aria-busy");
        textarea.focus();
      });
  });

  /* ------------------------------------------------------------------ *
   * Voice messages (Task 07).
   * ------------------------------------------------------------------ */

  var voiceBox = document.getElementById("voice-compose");
  var recordButton = document.getElementById("voice-record");
  var stopButton = document.getElementById("voice-stop");
  var fileInput = document.getElementById("voice-file");
  var voiceStatus = document.getElementById("voice-status");
  var voiceError = document.getElementById("voice-error");
  var recordingLabel = document.getElementById("voice-recording");
  var recordingElapsed = document.getElementById("voice-elapsed");

  if (!voiceBox || !recordButton || !stopButton || !fileInput ||
      !voiceStatus || !voiceError || !recordingLabel || !recordingElapsed) {
    return;
  }

  var maxAudioMb = Number(voiceBox.dataset.maxAudioMb || 0);
  var uploadInFlight = false;
  var recorder = null;
  var liveStream = null;
  var chunks = [];
  var preferredMimeTypes = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
  ];

  function showVoiceStatus(message) {
    voiceStatus.textContent = message;
    voiceStatus.hidden = false;
  }

  function hideVoiceStatus() {
    voiceStatus.textContent = "";
    voiceStatus.hidden = true;
  }

  function showVoiceError(message) {
    voiceError.textContent = message;
    voiceError.hidden = false;
  }

  function hideVoiceError() {
    voiceError.textContent = "";
    voiceError.hidden = true;
  }

  /* Recording duration indicator (T14).
   *
   * Invariants (keep them when editing this block):
   *  - At most one interval handle exists, ever. It is held in
   *    ``recordingTickerId``, created only by showRecordingLabel() (which
   *    clears any previous one first) and released by clearRecordingTicker().
   *    Every recording exit path goes through hideRecordingLabel().
   *  - The displayed value is recomputed from performance.now() on every
   *    tick, never accumulated from tick counts, so a throttled interval in
   *    a background tab cannot make the timer lag behind.
   *  - ``recordingLabel.hidden = true`` is only ever set by
   *    hideRecordingLabel(), which also stops the ticker and empties the
   *    text: a hidden indicator can never keep a timer or show a stale value.
   */
  var RECORDING_TICK_MS = 500;
  var recordingStartedAt = 0;
  var recordingTickerId = null;

  function padTwoDigits(value) {
    return value < 10 ? "0" + value : String(value);
  }

  /* "м:сс", and "ч:мм:сс" once the recording passes an hour. Never NaN,
     never negative, never empty. */
  function formatDuration(milliseconds) {
    var elapsed = Number(milliseconds);
    if (!isFinite(elapsed) || elapsed < 0) {
      elapsed = 0;
    }
    var totalSeconds = Math.floor(elapsed / 1000);
    var seconds = totalSeconds % 60;
    var minutes = Math.floor(totalSeconds / 60) % 60;
    var hours = Math.floor(totalSeconds / 3600);
    if (hours > 0) {
      return hours + ":" + padTwoDigits(minutes) + ":" + padTwoDigits(seconds);
    }
    return minutes + ":" + padTwoDigits(seconds);
  }

  function renderRecordingTime() {
    recordingElapsed.textContent =
      formatDuration(window.performance.now() - recordingStartedAt);
  }

  function clearRecordingTicker() {
    if (recordingTickerId !== null) {
      window.clearInterval(recordingTickerId);
      recordingTickerId = null;
    }
  }

  function hideRecordingLabel() {
    clearRecordingTicker();
    recordingElapsed.textContent = "";
    recordingLabel.hidden = true;
  }

  function showRecordingLabel() {
    clearRecordingTicker(); // never inherit a previous cycle's interval
    recordingStartedAt = window.performance.now();
    recordingElapsed.textContent = formatDuration(0);
    recordingLabel.hidden = false;
    recordingTickerId = window.setInterval(renderRecordingTime, RECORDING_TICK_MS);
  }

  function setVoiceControlsDisabled(disabled) {
    recordButton.disabled = disabled;
    stopButton.disabled = disabled;
    fileInput.disabled = disabled;
  }

  function recorderSupported() {
    return Boolean(
      window.MediaRecorder &&
      navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia
    );
  }

  function stopTracks(activeStream) {
    if (!activeStream) {
      return;
    }
    activeStream.getTracks().forEach(function (track) {
      track.stop();
    });
  }

  function uploadAudio(blob, fileName) {
    if (uploadInFlight) {
      return;
    }
    uploadInFlight = true;
    hideVoiceError();
    showVoiceStatus("Отправка…");
    setVoiceControlsDisabled(true);

    var formData = new FormData();
    formData.append("file", blob, fileName);

    fetch("/api/v1/audio", {
      method: "POST",
      body: formData, // no Content-Type header: the boundary is set by fetch
    })
      .then(function (response) {
        if (response.status === 201) {
          return response.json().then(function (created) {
            hideVoiceStatus();
            appendAudioMessage(created);
          });
        }
        // Static messages keyed on the status; the server payload is never
        // rendered into the page.
        if (response.status === 413) {
          throw { userMessage: "Файл слишком большой — не более " + maxAudioMb + " МБ." };
        }
        if (response.status === 400 || response.status === 422) {
          throw { userMessage: "Не удалось загрузить: неподдерживаемый формат или повреждённый файл." };
        }
        throw { userMessage: "Не удалось загрузить запись. Попробуйте ещё раз." };
      })
      .catch(function (error) {
        var message = (error && error.userMessage) ||
          "Не удалось загрузить запись. Попробуйте ещё раз.";
        showVoiceError(message);
      })
      .finally(function () {
        uploadInFlight = false;
        setVoiceControlsDisabled(false);
      });
  }

  function appendAudioMessage(payload) {
    if (emptyState && emptyState.parentNode === messageList) {
      emptyState.remove();
    }

    var article = document.createElement("article");
    article.className = "message message--audio";
    article.id = "message-" + payload.message_id;

    var meta = messageFooter(payload);
    var badge = document.createElement("span");
    badge.className = "message__badge";
    badge.textContent = "Голосовое";
    meta.appendChild(badge);

    var box = document.createElement("div");
    box.className = "message__audio";

    var player = document.createElement("audio");
    player.controls = true;
    player.preload = "metadata";
    // Server-generated playback URL derived from the response id only.
    player.src = "/api/v1/audio/" + payload.id + "/file";
    box.appendChild(player);

    // Transcription panel (Task 09): mirrors the server-side _message.html
    // structure so the transcribe flow below works on fresh uploads too.
    var panel = document.createElement("div");
    panel.className = "message__transcription";
    panel.id = "transcription-" + payload.id;
    panel.dataset.audioFileId = payload.id;
    panel.dataset.messageId = payload.message_id;
    panel.dataset.transcriptionState = "not_requested";

    var caption = document.createElement("p");
    caption.className = "message__caption";
    var hiddenLabel = document.createElement("span");
    hiddenLabel.className = "visually-hidden";
    hiddenLabel.textContent = "Статус расшифровки:";
    var statusText = document.createElement("span");
    statusText.setAttribute("data-role", "status-text");
    statusText.textContent = "Не расшифровано"; // manual STT starts only on click
    caption.appendChild(hiddenLabel);
    caption.appendChild(statusText);

    var raw = document.createElement("div");
    raw.className = "message__raw";
    raw.setAttribute("data-role", "raw");
    raw.hidden = true;

    var button = document.createElement("button");
    button.type = "button";
    button.className = "transcribe-button";
    button.setAttribute("data-action", "transcribe");
    button.textContent = "Расшифровать";

    var note = document.createElement("p");
    note.className = "message__note";
    note.setAttribute("data-role", "note");
    note.hidden = true;

    panel.appendChild(caption);
    panel.appendChild(raw);
    panel.appendChild(button);
    panel.appendChild(note);

    article.appendChild(meta);
    article.appendChild(box);
    article.appendChild(panel);
    article.appendChild(buildMessageActions(payload.message_id, payload.created_at));
    messageList.appendChild(article);
    article.scrollIntoView({ block: "nearest" });
  }

  function pickMimeType() {
    for (var index = 0; index < preferredMimeTypes.length; index += 1) {
      if (window.MediaRecorder.isTypeSupported(preferredMimeTypes[index])) {
        return preferredMimeTypes[index];
      }
    }
    return "";
  }

  if (!recorderSupported()) {
    // Upload-only fallback: keep the file control, drop the record button.
    recordButton.hidden = true;
  }
  voiceBox.hidden = false; // JS (and at least one input path) is available.

  recordButton.addEventListener("click", function () {
    if (recorder) {
      return; // already recording
    }
    hideVoiceError();
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then(function (media) {
        liveStream = media;
        chunks = [];
        var mimeType = pickMimeType();
        var options = {};
        if (mimeType) {
          options.mimeType = mimeType;
        }
        try {
          recorder = new MediaRecorder(media, options);
        } catch (exception) {
          recorder = null;
          stopTracks(liveStream);
          liveStream = null;
          showVoiceError("Запись недоступна в этом браузере — загрузите файл.");
          return;
        }
        recorder.addEventListener("dataavailable", function (event) {
          if (event.data && event.data.size > 0) {
            chunks.push(event.data);
          }
        });
        recorder.addEventListener("stop", function () {
          var finishedType = recorder.mimeType || mimeType || "audio/webm";
          var finishedStream = liveStream;
          recorder = null;
          liveStream = null;
          stopTracks(finishedStream);
          recordButton.hidden = false;
          stopButton.hidden = true;
          hideRecordingLabel();
          if (!chunks.length) {
            showVoiceError("Запись пуста. Попробуйте ещё раз.");
            chunks = [];
            return;
          }
          var blob = new Blob(chunks, { type: finishedType });
          chunks = [];
          var fileName = finishedType.indexOf("ogg") !== -1
            ? "recording.ogg"
            : "recording.webm";
          uploadAudio(blob, fileName);
        });
        // A recorder failure must not leave the duration ticking; the stop
        // event that follows it finishes the UI state.
        recorder.addEventListener("error", function () {
          hideRecordingLabel();
        });
        recorder.start();
        recordButton.hidden = true;
        stopButton.hidden = false;
        showRecordingLabel();
      })
      .catch(function () {
        hideRecordingLabel();
        showVoiceError("Микрофон недоступен или доступ запрещён. Можно загрузить файл.");
      });
  });

  stopButton.addEventListener("click", function () {
    if (recorder && recorder.state !== "inactive") {
      // The user's own stop is an exit path of its own: the duration is not
      // left depending on the recorder's "stop" event alone.
      hideRecordingLabel();
      recorder.stop();
    }
  });

  fileInput.addEventListener("change", function () {
    var file = fileInput.files && fileInput.files[0];
    if (!file) {
      return;
    }
    // The user-chosen file name goes only into the multipart part label; the
    // server never uses it (storage is under a server-generated UUID).
    uploadAudio(file, file.name);
    fileInput.value = "";
  });

  // Leaving the page (including the bfcache) must not strand the ticker.
  window.addEventListener("pagehide", function () {
    hideRecordingLabel();
  });
})();

/* ------------------------------------------------------------------ *
 * Manual transcription flow (Task 09).
 * ------------------------------------------------------------------ *
 * Clicking "Расшифровать"/"Перерасшифровать" under a voice message sends
 * POST /api/v1/audio/{id}/transcribe (a durable job — never STT in the page
 * request), then polls GET /api/v1/jobs/{id} until the job is terminal. A
 * succeeded job refreshes the raw text from GET .../transcriptions. Panels
 * rendered with an in-flight job (data-active-job-id, e.g. after a page
 * reload during processing) resume polling automatically. Raw transcription
 * text is always inserted via textContent — never markup.
 */
(function () {
  var STATUS_LABELS = {
    not_requested: "Не расшифровано",
    queued: "Ожидает расшифровки",
    running: "Расшифровывается…",
    succeeded: "Расшифровано",
    failed: "Ошибка расшифровки"
  };
  var POLL_INTERVAL_MS = 2000;
  var POLL_LIMIT = 450; // poll for ~15 minutes at most, then let the user retry

  function transcriptionUrl(audioFileId, suffix) {
    return "/api/v1/audio/" + encodeURIComponent(audioFileId) + suffix;
  }

  function hasSuccessFlag(panel) {
    return panel.getAttribute("data-has-success") === "true";
  }

  function setStatus(panel, status) {
    panel.setAttribute("data-transcription-state", status);
    var text = panel.querySelector('[data-role="status-text"]');
    if (text) {
      text.textContent = STATUS_LABELS[status] || STATUS_LABELS.not_requested;
    }
  }

  function setRawText(panel, text) {
    var raw = panel.querySelector('[data-role="raw"]');
    if (!raw) {
      return;
    }
    raw.textContent = text; // transcription text: textContent only
    raw.hidden = text.length === 0;
  }

  function showNote(panel, text) {
    var note = panel.querySelector('[data-role="note"]');
    if (!note) {
      return;
    }
    note.textContent = text;
    note.hidden = text.length === 0;
  }

  function setBusy(panel, busy) {
    var button = panel.querySelector('[data-action="transcribe"]');
    if (!button) {
      return;
    }
    if (busy) {
      button.disabled = true;
      button.textContent = "Расшифровка…";
    } else {
      button.disabled = false;
      button.textContent = hasSuccessFlag(panel) ? "Перерасшифровать" : "Расшифровать";
    }
  }

  function latestSucceededText(rows) {
    // The history endpoint lists runs oldest first; the newest success wins.
    for (var index = rows.length - 1; index >= 0; index -= 1) {
      if (rows[index].status === "succeeded") {
        return rows[index].raw_text || "";
      }
    }
    return null;
  }

  function lastRowStatus(rows) {
    if (!rows.length) {
      return "not_requested";
    }
    return rows[rows.length - 1].status;
  }

  function fetchHistory(panel) {
    return fetch(transcriptionUrl(panel.getAttribute("data-audio-file-id"), "/transcriptions"))
      .then(function (response) {
        if (!response.ok) {
          throw new Error("transcription-history-failed");
        }
        return response.json();
      });
  }

  function finishSucceeded(panel) {
    fetchHistory(panel)
      .then(function (rows) {
        var text = latestSucceededText(rows);
        if (text !== null) {
          panel.setAttribute("data-has-success", "true");
          setRawText(panel, text);
        }
        setStatus(panel, "succeeded");
      })
      .catch(function () {
        // History hiccup: still stop polling; a reload renders the server truth.
        setStatus(panel, "succeeded");
      })
      .finally(function () {
        panel.removeAttribute("data-active-job-id");
        setBusy(panel, false);
        showNote(panel, "");
      });
  }

  function finishFailed(panel) {
    setStatus(panel, "failed");
    // Keep any earlier raw text visible; learn whether a success ever existed
    // so the re-enabled button offers "Перерасшифровать".
    fetchHistory(panel)
      .then(function (rows) {
        if (latestSucceededText(rows) !== null) {
          panel.setAttribute("data-has-success", "true");
        }
      })
      .catch(function () {
        /* keep current state */
      })
      .finally(function () {
        panel.removeAttribute("data-active-job-id");
        setBusy(panel, false);
        showNote(panel, "");
      });
  }

  function restoreFromHistory(panel) {
    fetchHistory(panel)
      .then(function (rows) {
        var text = latestSucceededText(rows);
        if (text !== null) {
          panel.setAttribute("data-has-success", "true");
          setRawText(panel, text);
          setStatus(panel, "succeeded");
        } else if (rows.length) {
          setStatus(panel, lastRowStatus(rows));
        } else {
          setStatus(panel, "not_requested");
        }
      })
      .catch(function () {
        /* leave the current label untouched */
      });
  }

  function pollJob(panel, jobId) {
    panel.setAttribute("data-active-job-id", jobId);
    setStatus(panel, "queued");
    setBusy(panel, true);
    showNote(panel, "");

    var remaining = POLL_LIMIT;
    function schedule() {
      window.setTimeout(tick, POLL_INTERVAL_MS);
    }
    function giveUp() {
      panel.removeAttribute("data-active-job-id");
      setBusy(panel, false);
      showNote(panel, "Статус недоступен. Обновите страницу или попробуйте ещё раз.");
    }
    function tick() {
      remaining -= 1;
      if (remaining <= 0) {
        giveUp();
        return;
      }
      fetch("/api/v1/jobs/" + encodeURIComponent(jobId))
        .then(function (response) {
          if (!response.ok) {
            throw new Error("job-poll-failed");
          }
          return response.json();
        })
        .then(function (job) {
          if (job.status === "pending" || job.status === "retrying") {
            setStatus(panel, "queued");
            schedule();
          } else if (job.status === "running") {
            setStatus(panel, "running");
            schedule();
          } else if (job.status === "succeeded") {
            finishSucceeded(panel);
          } else {
            finishFailed(panel); // failed / cancelled / unknown terminal state
          }
        })
        .catch(function () {
          schedule(); // transient network error: keep polling
        });
    }
    tick();
  }

  function startTranscription(panel) {
    var state = panel.getAttribute("data-transcription-state");
    if (state === "queued" || state === "running" || panel.getAttribute("data-active-job-id")) {
      return; // a job is already being tracked
    }
    setBusy(panel, true);
    setStatus(panel, "queued");
    showNote(panel, "Отправка…");
    var audioFileId = panel.getAttribute("data-audio-file-id");
    fetch(transcriptionUrl(audioFileId, "/transcribe"), { method: "POST" })
      .then(function (response) {
        if (response.status === 202 || response.status === 200) {
          return response.json();
        }
        throw new Error("transcribe-request-failed");
      })
      .then(function (job) {
        showNote(panel, "");
        pollJob(panel, job.id);
      })
      .catch(function () {
        setBusy(panel, false);
        showNote(panel, "Не удалось запустить расшифровку. Попробуйте ещё раз.");
        restoreFromHistory(panel);
      });
  }

  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!target || typeof target.closest !== "function") {
      return;
    }
    var button = target.closest('[data-action="transcribe"]');
    if (!button || button.disabled) {
      return;
    }
    var panel = button.closest(".message__transcription");
    if (panel) {
      startTranscription(panel);
    }
  });

  function resumeActiveJobs() {
    var panels = document.querySelectorAll(".message__transcription[data-active-job-id]");
    for (var index = 0; index < panels.length; index += 1) {
      var jobId = panels[index].getAttribute("data-active-job-id");
      if (jobId) {
        pollJob(panels[index], jobId);
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", resumeActiveJobs);
  } else {
    resumeActiveJobs();
  }
})();

/* ------------------------------------------------------------------ *
 * Wellbeing run flow (Task 15).
 * ------------------------------------------------------------------ *
 * "Запустить wellbeing-рефлексию" reveals a confirmation panel whose submit
 * button stays disabled until the explicit checkbox ("Я понимаю: это не
 * диагноз…") is checked. Only then is POST /api/v1/pipelines/wellbeing/runs
 * sent with {export_id, confirm: true}; the server enforces confirm again
 * (double protection — AGENTS.md §16). After a 202 the job is polled like the
 * transcription flow and the page reloads on a terminal outcome. Error and
 * status strings are static; server payloads are never rendered.
 */
(function () {
  var form = document.getElementById("wellbeing-run-form");
  if (!form) {
    return;
  }
  var exportId = form.getAttribute("data-export-id") || "";
  var openButton = document.getElementById("wellbeing-run-open");
  var confirmPanel = document.getElementById("wellbeing-confirm");
  var confirmCheck = document.getElementById("wellbeing-confirm-check");
  var submitButton = document.getElementById("wellbeing-run-submit");
  var errorBox = document.getElementById("wellbeing-run-error");

  if (!exportId || !openButton || !confirmPanel || !confirmCheck ||
      !submitButton || !errorBox) {
    return;
  }

  function showRunError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function setBusy(busy) {
    openButton.disabled = busy;
    submitButton.disabled = busy || !confirmCheck.checked;
    if (busy) {
      submitButton.textContent = "Запуск…";
    } else {
      submitButton.textContent = "Подтверждаю и запускаю";
    }
  }

  openButton.addEventListener("click", function () {
    confirmPanel.hidden = false;
    openButton.hidden = true;
  });

  confirmCheck.addEventListener("change", function () {
    submitButton.disabled = !confirmCheck.checked;
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    if (!confirmCheck.checked) {
      showRunError("Подтвердите, что вы понимаете назначение рефлексии.");
      return;
    }
    errorBox.hidden = true;
    setBusy(true);
    fetch("/api/v1/pipelines/wellbeing/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ export_id: exportId, confirm: true }),
    })
      .then(function (response) {
        if (response.status === 202) {
          return response.json();
        }
        if (response.status === 403) {
          throw { userMessage: "Wellbeing-анализ отключён в настройках приложения." };
        }
        if (response.status === 409) {
          throw { userMessage: "Требуется явное подтверждение запуска." };
        }
        if (response.status === 404) {
          throw { userMessage: "Сначала выполните нормализацию этой выгрузки." };
        }
        throw { userMessage: "Не удалось запустить анализ. Попробуйте ещё раз." };
      })
      .then(function (job) {
        pollWellbeingJob(job.id);
      })
      .catch(function (error) {
        var message = (error && error.userMessage) ||
          "Не удалось запустить анализ. Попробуйте ещё раз.";
        showRunError(message);
        setBusy(false);
      });
  });

  function pollWellbeingJob(jobId) {
    var attempts = 0;
    function tick() {
      attempts += 1;
      if (attempts > 450) {
        showRunError("Анализ выполняется. Обновите страницу позже.");
        setBusy(false);
        return;
      }
      fetch("/api/v1/jobs/" + encodeURIComponent(jobId))
        .then(function (response) {
          if (!response.ok) {
            throw new Error("job-poll-failed");
          }
          return response.json();
        })
        .then(function (job) {
          if (job.status === "pending" || job.status === "running" ||
              job.status === "retrying") {
            window.setTimeout(tick, 2000);
          } else {
            window.setTimeout(function () {
              window.location.reload();
            }, 300);
          }
        })
        .catch(function () {
          window.setTimeout(tick, 2000);
        });
    }
    tick();
  }
})();

/* ------------------------------------------------------------------ *
 * Message deletion (task T15, ADR-011).
 * ------------------------------------------------------------------ *
 * "Удалить" under a message does not delete anything by itself: it only
 * reveals the inline confirmation panel. The deletion happens after the
 * explicit "Да, удалить" (or never, on "Отмена"/Escape) because it is
 * irreversible — the message text and, for a voice message, the stored audio
 * file are erased. The request is the existing
 * DELETE /api/v1/messages/{id}: 204 and 404 both mean "the message is gone"
 * (404 = unknown or already deleted) and both remove it from the list without
 * a page reload. Click handling is delegated from the document, so messages
 * appended by this file and messages loaded later through htmx work the same.
 * Status and error strings are static; the server payload is never rendered.
 */
(function () {
  var GENERIC_ERROR = "Не удалось удалить запись. Попробуйте ещё раз.";
  var DELETING = "Удаление…";
  var EMPTY_STATE_TEXT = "Пока нет записей. Напишите первую ниже.";

  function deleteButtonFor(article) {
    return article.querySelector('[data-action="delete-message"]');
  }

  function confirmPanelFor(article) {
    return article.querySelector('[data-role="delete-confirm"]');
  }

  function setStatus(article, text) {
    var status = article.querySelector('[data-role="delete-status"]');
    if (!status) {
      return;
    }
    status.textContent = text;
    status.hidden = text.length === 0;
  }

  function setPanelBusy(article, busy) {
    var panel = confirmPanelFor(article);
    if (!panel) {
      return;
    }
    var buttons = panel.querySelectorAll("button");
    for (var index = 0; index < buttons.length; index += 1) {
      buttons[index].disabled = busy;
    }
  }

  function openConfirm(article) {
    var panel = confirmPanelFor(article);
    var open = deleteButtonFor(article);
    if (!panel || !open) {
      return;
    }
    setStatus(article, "");
    setPanelBusy(article, false);
    open.hidden = true;
    panel.hidden = false;
    var confirmButton = panel.querySelector('[data-action="delete-message-confirm"]');
    if (confirmButton) {
      confirmButton.focus(); // the second step must not be reachable by accident
    }
  }

  function closeConfirm(article) {
    var panel = confirmPanelFor(article);
    var open = deleteButtonFor(article);
    if (!panel || !open) {
      return;
    }
    panel.hidden = true;
    open.hidden = false;
    setStatus(article, "");
    open.focus();
  }

  function ensureEmptyState(messageList) {
    if (messageList.querySelector(".message") !== null) {
      return;
    }
    if (document.getElementById("older-control") !== null) {
      // Older messages are still paginated above the list: "no entries yet"
      // would be a lie.
      return;
    }
    var empty = document.createElement("p");
    empty.id = "message-empty";
    empty.className = "message-list__empty";
    empty.textContent = EMPTY_STATE_TEXT;
    messageList.appendChild(empty);
  }

  function messageIdOf(article) {
    var button = deleteButtonFor(article);
    var declared = button ? button.getAttribute("data-message-id") : null;
    if (declared) {
      return declared;
    }
    var prefix = "message-"; // server-rendered article id is "message-<uuid>"
    return article.id.indexOf(prefix) === 0 ? article.id.slice(prefix.length) : "";
  }

  function focusAfterRemoval(neighbour) {
    var button = neighbour ? deleteButtonFor(neighbour) : null;
    if (button) {
      button.focus();
      return;
    }
    var textarea = document.getElementById("message-text");
    if (textarea) {
      textarea.focus();
    }
  }

  function removeMessage(article) {
    var messageList = article.parentNode;
    // Remember a neighbour before the article (and the focused button) goes:
    // the keyboard user must not be dropped back to the top of the document.
    var neighbour = article.nextElementSibling || article.previousElementSibling;
    article.remove();
    if (messageList && messageList.id === "message-list") {
      ensureEmptyState(messageList);
    }
    focusAfterRemoval(neighbour);
  }

  function confirmDelete(article) {
    var messageId = messageIdOf(article);
    if (!messageId) {
      setStatus(article, GENERIC_ERROR);
      return;
    }
    setPanelBusy(article, true);
    setStatus(article, DELETING);
    fetch("/api/v1/messages/" + encodeURIComponent(messageId), { method: "DELETE" })
      .then(function (response) {
        if (response.status === 204 || response.status === 404) {
          // 204: deleted. 404: unknown or already deleted — same outcome for
          // the user, so it is not treated as a failure.
          removeMessage(article);
          return;
        }
        throw new Error("delete-failed");
      })
      .catch(function () {
        setPanelBusy(article, false);
        setStatus(article, GENERIC_ERROR);
      });
  }

  document.addEventListener("click", function (event) {
    var target = event.target;
    if (!target || typeof target.closest !== "function") {
      return;
    }
    var article = target.closest(".message");
    if (!article) {
      return;
    }
    if (target.closest('[data-action="delete-message"]')) {
      openConfirm(article);
      return;
    }
    if (target.closest('[data-action="delete-message-confirm"]')) {
      confirmDelete(article);
      return;
    }
    if (target.closest('[data-action="delete-message-cancel"]')) {
      closeConfirm(article);
    }
  });

  document.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") {
      return;
    }
    var target = event.target;
    if (!target || typeof target.closest !== "function") {
      return;
    }
    var article = target.closest(".message");
    if (!article) {
      return;
    }
    var panel = confirmPanelFor(article);
    if (panel && !panel.hidden) {
      closeConfirm(article);
    }
  });
})();

/* ------------------------------------------------------------------ *
 * Theme switch (task T16a).
 * ------------------------------------------------------------------ *
 * The *theme* itself is applied before the first paint by the inline script
 * in base.html (a stored choice is written to data-theme synchronously, so a
 * dark page never flashes light). This module owns the parts that need the
 * DOM:
 *
 *  - it shows the current state ("Светлая"/"Тёмная") in every toggle, and —
 *    because the label lives in an aria-live region — announces the change
 *    to screen readers;
 *  - clicking a toggle writes the opposite theme to data-theme and remembers
 *    it in localStorage (one key, the theme name only: no diary content);
 *  - while the user has made no explicit choice nothing is stored and the
 *    CSS prefers-color-scheme block keeps following the system setting, so
 *    the label follows it too.
 *
 * There are two toggles (sidebar and, at narrow widths, the topbar); both
 * carry data-action="toggle-theme" and share this handler, and only one of
 * them is ever visible (see the shell CSS).
 */
(function () {
  var root = document.documentElement;
  var STORAGE_KEY = "mydiary-theme";
  var DARK = "dark";
  var LIGHT = "light";

  var toggles = document.querySelectorAll('[data-action="toggle-theme"]');
  var labels = document.querySelectorAll('[data-role="theme-name"]');
  if (!toggles.length) {
    return;
  }

  function systemTheme() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? DARK
      : LIGHT;
  }

  function storedTheme() {
    try {
      var value = window.localStorage.getItem(STORAGE_KEY);
      return value === DARK || value === LIGHT ? value : null;
    } catch (error) {
      return null; // storage blocked: the system setting stays in charge
    }
  }

  function currentTheme() {
    var attribute = root.getAttribute("data-theme");
    if (attribute === DARK || attribute === LIGHT) {
      return attribute;
    }
    return systemTheme();
  }

  function renderLabels() {
    var label = currentTheme() === DARK ? "Тёмная" : "Светлая";
    for (var index = 0; index < labels.length; index += 1) {
      labels[index].textContent = label;
    }
  }

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    renderLabels();
  }

  for (var index = 0; index < toggles.length; index += 1) {
    toggles[index].addEventListener("click", function () {
      var next = currentTheme() === DARK ? LIGHT : DARK;
      try {
        window.localStorage.setItem(STORAGE_KEY, next);
      } catch (error) {
        /* Storage blocked: the choice still applies to this page. */
      }
      applyTheme(next);
    });
  }

  renderLabels();

  // No explicit choice yet: keep the label in step with the system setting.
  if (storedTheme() === null && window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      if (storedTheme() === null) {
        renderLabels();
      }
    });
  }
})();
