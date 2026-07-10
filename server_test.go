package main

import (
	"strings"
	"testing"
)

// countSystemMessages returns how many top-level messages have role "system".
func countSystemMessages(messages []any) int {
	count := 0
	for _, raw := range messages {
		msg, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(toString(msg["role"])), "system") {
			count++
		}
	}
	return count
}

func toString(value any) string {
	if s, ok := value.(string); ok {
		return s
	}
	return ""
}

func TestEnsureFreebuffSystemMarkerMergesIntoStringSystemMessage(t *testing.T) {
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "You are Koda, the architect."},
			map[string]any{"role": "user", "content": "hello"},
		},
	}

	ensureFreebuffSystemMarker(payload)

	messages := payload["messages"].([]any)
	if got := countSystemMessages(messages); got != 1 {
		t.Fatalf("expected exactly 1 system message, got %d", got)
	}
	system := messages[0].(map[string]any)
	content, ok := system["content"].(string)
	if !ok {
		t.Fatalf("expected string system content, got %T", system["content"])
	}
	if !strings.HasPrefix(content, freebuffSystemMarker) {
		t.Errorf("expected system content to start with marker %q, got %q", freebuffSystemMarker, content)
	}
	if !strings.Contains(content, "You are Koda, the architect.") {
		t.Errorf("expected original system text preserved, got %q", content)
	}
}

func TestEnsureFreebuffSystemMarkerMergesIntoArraySystemMessage(t *testing.T) {
	payload := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "system",
				"content": []any{
					map[string]any{"type": "text", "text": "You are Koda."},
				},
			},
			map[string]any{"role": "user", "content": "hello"},
		},
	}

	ensureFreebuffSystemMarker(payload)

	messages := payload["messages"].([]any)
	if got := countSystemMessages(messages); got != 1 {
		t.Fatalf("expected exactly 1 system message, got %d", got)
	}
	parts, ok := messages[0].(map[string]any)["content"].([]any)
	if !ok {
		t.Fatalf("expected array system content, got %T", messages[0].(map[string]any)["content"])
	}
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts (marker + original), got %d", len(parts))
	}
	first := parts[0].(map[string]any)
	if first["type"] != "text" || first["text"] != freebuffSystemMarker {
		t.Errorf("expected first part to be the marker, got %#v", first)
	}
}

func TestEnsureFreebuffSystemMarkerPrependsStandaloneWhenAbsent(t *testing.T) {
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hello"},
		},
	}

	ensureFreebuffSystemMarker(payload)

	messages := payload["messages"].([]any)
	if got := countSystemMessages(messages); got != 1 {
		t.Fatalf("expected exactly 1 system message, got %d", got)
	}
	first := messages[0].(map[string]any)
	if first["role"] != "system" {
		t.Fatalf("expected first message to be system, got role %v", first["role"])
	}
	if first["content"] != freebuffSystemMarker {
		t.Errorf("expected standalone marker content, got %v", first["content"])
	}
}

func TestEnsureFreebuffSystemMarkerNoopWhenMarkerPresent(t *testing.T) {
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "You are Buffy the agent."},
			map[string]any{"role": "user", "content": "hello"},
		},
	}

	ensureFreebuffSystemMarker(payload)

	messages := payload["messages"].([]any)
	if got := countSystemMessages(messages); got != 1 {
		t.Fatalf("expected exactly 1 system message, got %d", got)
	}
	if content := messages[0].(map[string]any)["content"]; content != "You are Buffy the agent." {
		t.Errorf("expected content unchanged, got %v", content)
	}
}

func TestEnsureFreebuffSystemMarkerNoopWhenNoMessages(t *testing.T) {
	payload := map[string]any{}
	ensureFreebuffSystemMarker(payload)
	if _, ok := payload["messages"]; ok {
		t.Errorf("expected no messages key to be created")
	}
}
