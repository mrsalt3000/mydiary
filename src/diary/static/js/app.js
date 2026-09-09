/*
 * Local Diary — chat page behaviour (Task 06).
 *
 * Scope: compose form only. The "load older messages" paging is handled
 * declaratively by htmx (server-rendered fragments + out-of-band swap of the
 * "load older" control), so this file stays minimal and dependency-free.
 *
 * Safety rules (must be preserved by any later edit):
 *  - Diary text is NEVER placed into the DOM as markup. The new message is
 *    built with createElement + textContent only — no HTML-string sinks.
 *  - Error and info strings shown to the user are static; server-provided
 *    error payloads are never echoed into the page.
 *  - No fetch/script/link to external hosts: the page only talks to this app.
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

  function appendMessage(message) {
    if (emptyState && emptyState.parentNode === messageList) {
      emptyState.remove();
    }

    var article = document.createElement("article");
    article.className = "message";

    var meta = document.createElement("header");
    meta.className = "message__meta";
    var time = document.createElement("time");
    time.dateTime = message.created_at;
    time.textContent = formatTimestamp(message.created_at);
    meta.appendChild(time);

    var body = document.createElement("div");
    body.className = "message__body";
    body.textContent = message.body; // diary text: textContent only, never markup

    article.appendChild(meta);
    article.appendChild(body);
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
})();
