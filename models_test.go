package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"testing"
	"time"
)

func TestParseConstantsDirectStrings(t *testing.T) {
	source := `
export const FREEBUFF_DEEPSEEK_V4_PRO_MODEL_ID = 'deepseek/deepseek-v4-pro'
export const FREEBUFF_DEEPSEEK_V4_FLASH_MODEL_ID = 'deepseek/deepseek-v4-flash'
export const FREEBUFF_KIMI_MODEL_ID = moonshotModels.kimiK26
export const FREEBUFF_MINIMAX_MODEL_ID = 'minimax/minimax-m2.7'
`
	constants := parseConstants(source, nil)
	if constants["FREEBUFF_DEEPSEEK_V4_PRO_MODEL_ID"] != "deepseek/deepseek-v4-pro" {
		t.Errorf("expected deepseek/deepseek-v4-pro, got %s", constants["FREEBUFF_DEEPSEEK_V4_PRO_MODEL_ID"])
	}
	if constants["FREEBUFF_DEEPSEEK_V4_FLASH_MODEL_ID"] != "deepseek/deepseek-v4-flash" {
		t.Errorf("expected deepseek/deepseek-v4-flash, got %s", constants["FREEBUFF_DEEPSEEK_V4_FLASH_MODEL_ID"])
	}
	// kimiK26 should not resolve yet (no object context)
	if constants["FREEBUFF_KIMI_MODEL_ID"] != "" {
		t.Errorf("expected empty for unresolved object ref, got %s", constants["FREEBUFF_KIMI_MODEL_ID"])
	}
}

func TestParseConstantsObjectDefs(t *testing.T) {
	// First parse model-config to populate object properties
	modelConfigSrc := `
export const moonshotModels = {
  kimiK26: 'moonshotai/kimi-k2.6',
  kimiOld: 'moonshotai/kimi-old',
}
export const minimaxModels = {
  minimaxM3: 'minimax/minimax-m3',
}
`
	constants := parseConstants(modelConfigSrc, nil)

	// Check object properties were stored
	if constants["moonshotModels.kimiK26"] != "moonshotai/kimi-k2.6" {
		t.Errorf("expected moonshotai/kimi-k2.6, got %s", constants["moonshotModels.kimiK26"])
	}
	if constants["minimaxModels.minimaxM3"] != "minimax/minimax-m3" {
		t.Errorf("expected minimax/minimax-m3, got %s", constants["minimaxModels.minimaxM3"])
	}

	// Now parse freebuff-models.ts which references the objects
	freebuffModelsSrc := `
export const FREEBUFF_KIMI_MODEL_ID = moonshotModels.kimiK26
export const FREEBUFF_MINIMAX_M3_MODEL_ID = minimaxModels.minimaxM3
export const FREEBUFF_DEEPSEEK_V4_PRO_MODEL_ID = 'deepseek/deepseek-v4-pro'
`
	constants = parseConstants(freebuffModelsSrc, constants)

	if constants["FREEBUFF_KIMI_MODEL_ID"] != "moonshotai/kimi-k2.6" {
		t.Errorf("expected moonshotai/kimi-k2.6, got %s", constants["FREEBUFF_KIMI_MODEL_ID"])
	}
	if constants["FREEBUFF_MINIMAX_M3_MODEL_ID"] != "minimax/minimax-m3" {
		t.Errorf("expected minimax/minimax-m3, got %s", constants["FREEBUFF_MINIMAX_M3_MODEL_ID"])
	}
	if constants["FREEBUFF_DEEPSEEK_V4_PRO_MODEL_ID"] != "deepseek/deepseek-v4-pro" {
		t.Errorf("expected deepseek/deepseek-v4-pro, got %s", constants["FREEBUFF_DEEPSEEK_V4_PRO_MODEL_ID"])
	}
}

func TestParseAllFreeModelsWithConstants(t *testing.T) {
	constants := map[string]string{
		"FREEBUFF_MINIMAX_MODEL_ID":         "minimax/minimax-m2.7",
		"FREEBUFF_MINIMAX_M3_MODEL_ID":      "minimax/minimax-m3",
		"FREEBUFF_KIMI_MODEL_ID":            "moonshotai/kimi-k2.6",
		"FREEBUFF_DEEPSEEK_V4_PRO_MODEL_ID": "deepseek/deepseek-v4-pro",
		"FREEBUFF_DEEPSEEK_V4_FLASH_MODEL_ID": "deepseek/deepseek-v4-flash",
		"FREEBUFF_MIMO_V25_MODEL_ID":        "mimo/mimo-v2.5",
		"FREEBUFF_MIMO_V25_PRO_MODEL_ID":    "mimo/mimo-v2.5-pro",
		"FREEBUFF_GLM_V52_MODEL_ID":         "z-ai/glm-5.2",
		"FREEBUFF_GEMINI_PRO_MODEL_ID":      "google/gemini-3.1-pro-preview",
	}

	source := `
export const FREE_MODE_AGENT_MODELS: Record<string, Set<string>> = {
  'base2-free': new Set([
    FREEBUFF_MINIMAX_MODEL_ID,
    FREEBUFF_MINIMAX_M3_MODEL_ID,
    FREEBUFF_DEEPSEEK_V4_PRO_MODEL_ID,
    FREEBUFF_DEEPSEEK_V4_FLASH_MODEL_ID,
    FREEBUFF_KIMI_MODEL_ID,
    FREEBUFF_MIMO_V25_PRO_MODEL_ID,
    FREEBUFF_MIMO_V25_MODEL_ID,
  ]),
  'file-picker': new Set(['google/gemini-2.5-flash-lite']),
  'code-reviewer-minimax': new Set([FREEBUFF_MINIMAX_MODEL_ID]),
  'code-reviewer-minimax-m3': new Set([FREEBUFF_MINIMAX_M3_MODEL_ID]),
}
`
	result := parseAllFreeModels(source, constants)

	base2free := result["base2-free"]
	if len(base2free) != 7 {
		t.Errorf("expected 7 models for base2-free, got %d: %v", len(base2free), base2free)
	}

	// Check specific models
	found := false
	for _, m := range base2free {
		if m == "minimax/minimax-m3" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected minimax/minimax-m3 in base2-free models, got %v", base2free)
	}

	found = false
	for _, m := range base2free {
		if m == "moonshotai/kimi-k2.6" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected moonshotai/kimi-k2.6 in base2-free models, got %v", base2free)
	}

	// file-picker uses literal string
	fp := result["file-picker"]
	if len(fp) != 1 || fp[0] != "google/gemini-2.5-flash-lite" {
		t.Errorf("expected [google/gemini-2.5-flash-lite] for file-picker, got %v", fp)
	}

	// code-reviewer-minimax should resolve FREEBUFF_MINIMAX_MODEL_ID
	crm := result["code-reviewer-minimax"]
	if len(crm) != 1 || crm[0] != "minimax/minimax-m2.7" {
		t.Errorf("expected [minimax/minimax-m2.7] for code-reviewer-minimax, got %v", crm)
	}
}

func TestLiveFetchAndParse(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("skipping live network test in CI")
	}

	client := &http.Client{Timeout: 15 * time.Second}
	ctx := context.Background()
	logger := log.New(os.Stderr, "[test] ", 0)

	registry := NewModelRegistry(client, logger)
	err := registry.refresh(ctx)
	if err != nil {
		t.Fatalf("refresh failed: %v", err)
	}

	models := registry.Models()
	if len(models) == 0 {
		t.Fatal("expected at least 1 model, got 0")
	}

	t.Logf("Discovered %d models: %v", len(models), models)

	// Verify key models are present
	expectedModels := []string{
		"minimax/minimax-m3",
		"minimax/minimax-m2.7",
		"deepseek/deepseek-v4-pro",
		"deepseek/deepseek-v4-flash",
		"moonshotai/kimi-k2.6",
		"mimo/mimo-v2.5-pro",
		"mimo/mimo-v2.5",
		"z-ai/glm-5.2",
	}
	for _, expected := range expectedModels {
		if !registry.HasModel(expected) {
			t.Errorf("expected model %s not found in registry", expected)
		}
	}

	// Verify agent mapping works
	for _, expected := range expectedModels {
		agent, ok := registry.AgentForModel(expected)
		if !ok {
			t.Errorf("no agent mapping for %s", expected)
		} else {
			t.Logf("model %s → agent %s", expected, agent)
		}
	}
}
