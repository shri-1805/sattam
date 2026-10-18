const form = document.querySelector("#chat-form");
const input = document.querySelector("#scenario-input");
const conversation = document.querySelector("#conversation");
const welcome = document.querySelector("#welcome");
const sendButton = document.querySelector("#send-button");
const characterCount = document.querySelector("#character-count");
const newChatButton = document.querySelector("#new-chat");
const MAX_SCENARIO_LENGTH = 12_000;
let isSending = false;

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function scrollToLatest() {
  window.requestAnimationFrame(() => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  });
}

function setSending(sending) {
  isSending = sending;
  input.disabled = sending;
  sendButton.disabled =
    sending || !input.value.trim() || input.value.length > MAX_SCENARIO_LENGTH;
  sendButton.setAttribute("aria-label", sending ? "Analyzing scenario" : "Send scenario");
  if (!sending) input.focus();
}

function addUserMessage(scenario) {
  const message = createElement("div", "message");
  const bubble = createElement("div", "user-message", scenario);
  bubble.setAttribute("aria-label", "Your scenario");
  message.append(bubble);
  conversation.append(message);
  scrollToLatest();
}

function addTypingIndicator() {
  const message = createElement("div", "message");
  message.id = "typing-message";
  const heading = createElement("div", "assistant-heading");
  const avatar = createAssistantAvatar();
  heading.append(avatar, createElement("span", "", "Finding relevant provisions"));
  const indicator = createElement("div", "typing-indicator");
  indicator.setAttribute("role", "status");
  indicator.setAttribute("aria-label", "Analyzing your scenario");
  for (let index = 0; index < 3; index += 1) {
    indicator.append(createElement("span"));
  }
  message.append(heading, indicator);
  conversation.append(message);
  scrollToLatest();
  return message;
}

function createAssistantAvatar() {
  const avatar = createElement("span", "assistant-avatar");
  avatar.setAttribute("aria-hidden", "true");
  avatar.innerHTML =
    '<svg viewBox="0 0 36 36" fill="none"><path d="M18 5v26M7 12h22M9.5 12 5 22h9L9.5 12Zm17 0-4.5 10h9l-4.5-10Z"/><path d="M12 30h12"/></svg>';
  return avatar;
}

function addError(messageText) {
  const message = createElement("div", "message");
  const heading = createElement("div", "assistant-heading");
  heading.append(createAssistantAvatar(), createElement("span", "", "Analysis unavailable"));
  const error = createElement("div", "error-message");
  error.setAttribute("role", "alert");
  error.append(
    createElement("strong", "", "I couldn’t complete that request."),
    document.createTextNode(messageText),
  );
  message.append(heading, error);
  conversation.append(message);
  scrollToLatest();
}

function addAnswer(result) {
  const message = createElement("div", "message");
  const heading = createElement("div", "assistant-heading");
  heading.append(createAssistantAvatar(), createElement("span", "", "Relevant provisions"));
  message.append(heading);

  if (!result.clauses?.length) {
    message.append(
      createElement(
        "div",
        "empty-answer",
        "I couldn’t identify a clearly applicable provision among the retrieved fundamental-rights clauses. This does not establish that no rights or legal remedies apply.",
      ),
    );
  } else {
    message.append(
      createElement(
        "p",
        "answer-intro",
        `I found ${result.clauses.length} provision${result.clauses.length === 1 ? "" : "s"} that may be relevant to the situation you described.`,
      ),
    );
    const list = createElement("div", "clause-list");
    for (const clause of result.clauses) {
      list.append(createClauseCard(clause));
    }
    message.append(list);
  }

  if (result.notice) {
    const notice = createElement("div", "legal-notice");
    notice.append(
      createElement("span", "", "ⓘ"),
      createElement("span", "", result.notice),
    );
    message.append(notice);
  }
  conversation.append(message);
  scrollToLatest();
}

function createClauseCard(clause) {
  const card = createElement("article", "clause-card");
  const header = createElement("div", "clause-card-header");
  const source = createElement("div", "clause-source");
  const reference = createElement(
    "span",
    "clause-reference",
    clause.reference || clause.article_id || clause.id,
  );
  if (clause.id && clause.id !== clause.article_id) {
    reference.append(document.createTextNode(` · ${clause.id}`));
  }
  source.append(
    reference,
    createElement("h2", "clause-title", clause.title),
  );
  const documentName = createElement("span", "clause-document", clause.document);
  header.append(source, documentName);

  const legalText = createElement("blockquote", "clause-text", clause.text);
  const applicability = createElement("div", "clause-applicability");
  applicability.append(
    createElement("strong", "", "Why it may apply"),
    document.createTextNode(clause.applicability),
  );
  card.append(header, legalText, applicability);

  if (clause.rights?.length) {
    const tags = createElement("div", "right-tags");
    for (const right of clause.rights) {
      tags.append(createElement("span", "right-tag", right));
    }
    card.append(tags);
  }
  return card;
}

async function submitScenario(scenario) {
  const trimmedScenario = scenario.trim();
  if (!trimmedScenario || isSending || trimmedScenario.length > MAX_SCENARIO_LENGTH) return;

  welcome?.remove();
  addUserMessage(trimmedScenario);
  input.value = "";
  updateComposer();
  setSending(true);
  const typing = addTypingIndicator();

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: trimmedScenario }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail =
        typeof body.detail === "string"
          ? body.detail
          : "Please check the service connection and try again.";
      throw new Error(detail);
    }
    addAnswer(body);
  } catch (error) {
    addError(error instanceof Error ? error.message : "An unexpected error occurred.");
  } finally {
    typing.remove();
    setSending(false);
  }
}

function updateComposer() {
  const length = input.value.length;
  characterCount.textContent = `${length.toLocaleString()} / 12,000`;
  sendButton.disabled = isSending || !input.value.trim() || length > MAX_SCENARIO_LENGTH;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void submitScenario(input.value);
});

input.addEventListener("input", updateComposer);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

conversation.addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) return;
  const button = event.target.closest("[data-prompt]");
  if (!button) return;
  input.value = button.dataset.prompt || "";
  updateComposer();
  input.focus();
  form.requestSubmit();
});

newChatButton.addEventListener("click", () => {
  if (isSending) return;
  conversation.replaceChildren(welcome.cloneNode(true));
  input.value = "";
  updateComposer();
  input.focus();
});

updateComposer();
