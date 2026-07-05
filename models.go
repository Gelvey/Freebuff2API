package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	freeAgentsSourceURL   = "https://raw.githubusercontent.com/CodebuffAI/codebuff/main/common/src/constants/free-agents.ts"
	freebuffModelsURL     = "https://raw.githubusercontent.com/CodebuffAI/codebuff/main/common/src/constants/freebuff-models.ts"
	modelConfigURL        = "https://raw.githubusercontent.com/CodebuffAI/codebuff/main/common/src/constants/model-config.ts"
	modelRefreshInterval  = 6 * time.Hour
)

// hardcodedFallback is used when the remote fetch fails on startup.
// Last updated: 2026-07-05 from CodebuffAI/codebuff freebuff-models.ts.
var hardcodedFallback = map[string][]string{
	"base2-free":         {"minimax/minimax-m3", "minimax/minimax-m2.7", "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "moonshotai/kimi-k2.6", "mimo/mimo-v2.5-pro", "mimo/mimo-v2.5", "z-ai/glm-5.2"},
	"file-picker":        {"google/gemini-2.5-flash-lite"},
	"file-picker-max":    {"google/gemini-3.1-flash-lite-preview"},
	"file-lister":        {"google/gemini-3.1-flash-lite-preview"},
	"researcher-web":     {"google/gemini-3.1-flash-lite-preview"},
	"researcher-docs":    {"google/gemini-3.1-flash-lite-preview"},
	"basher":             {"google/gemini-3.1-flash-lite-preview"},
	"editor-lite":        {"minimax/minimax-m3", "minimax/minimax-m2.7", "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "moonshotai/kimi-k2.6", "mimo/mimo-v2.5-pro", "mimo/mimo-v2.5", "z-ai/glm-5.2"},
	"code-reviewer-lite": {"minimax/minimax-m3", "minimax/minimax-m2.7", "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "moonshotai/kimi-k2.6", "mimo/mimo-v2.5-pro", "mimo/mimo-v2.5", "z-ai/glm-5.2"},
}

// ModelRegistry fetches and caches the agent→model mapping for all free agents
// from the upstream free-agents.ts source file.
type ModelRegistry struct {
	client *http.Client
	logger *log.Logger

	mu           sync.RWMutex
	agentModels  map[string][]string // agentID → []model
	modelToAgent map[string]string   // model → chosen agentID
	allModels    []string            // deduplicated, sorted
	lastOK       time.Time

	stopCh chan struct{}
	wg     sync.WaitGroup
}

func NewModelRegistry(client *http.Client, logger *log.Logger) *ModelRegistry {
	return &ModelRegistry{
		client:       client,
		logger:       logger,
		agentModels:  make(map[string][]string),
		modelToAgent: make(map[string]string),
		stopCh:       make(chan struct{}),
	}
}

func (r *ModelRegistry) Start(ctx context.Context) {
	if err := r.refresh(ctx); err != nil {
		r.logger.Printf("model registry: initial fetch failed, loading hardcoded fallback: %v", err)
		r.loadFallback()
	}

	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		ticker := time.NewTicker(modelRefreshInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				if err := r.refresh(ctx); err != nil {
					r.logger.Printf("model registry: refresh failed: %v", err)
				}
				cancel()
			case <-r.stopCh:
				return
			}
		}
	}()
}

func (r *ModelRegistry) Stop() {
	close(r.stopCh)
	r.wg.Wait()
}

// Models returns the deduplicated list of all available model names.
func (r *ModelRegistry) Models() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, len(r.allModels))
	copy(out, r.allModels)
	return out
}

// HasModel checks if the given model is available.
func (r *ModelRegistry) HasModel(model string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.modelToAgent[model]
	return ok
}

// AgentForModel returns the agent ID that should serve the given model.
func (r *ModelRegistry) AgentForModel(model string) (string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	agent, ok := r.modelToAgent[model]
	return agent, ok
}

// AgentIDs returns the list of all known agent IDs.
func (r *ModelRegistry) AgentIDs() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ids := make([]string, 0, len(r.agentModels))
	for id := range r.agentModels {
		ids = append(ids, id)
	}
	return ids
}

func (r *ModelRegistry) refresh(ctx context.Context) error {
	// Fetch all 3 source files in sequence (small files, single HTTP pipeline).
	// Order matters: model-config.ts → freebuff-models.ts → free-agents.ts
	// so that constants are fully resolved before parsing agent mappings.
	modelConfigSrc, err := fetchSource(ctx, r.client, modelConfigURL)
	if err != nil {
		r.logger.Printf("model registry: fetch model-config failed (non-fatal): %v", err)
	}

	freebuffModelsSrc, err := fetchSource(ctx, r.client, freebuffModelsURL)
	if err != nil {
		r.logger.Printf("model registry: fetch freebuff-models failed (non-fatal): %v", err)
	}

	freeAgentsSrc, err := fetchSource(ctx, r.client, freeAgentsSourceURL)
	if err != nil {
		return fmt.Errorf("fetch free-agents source: %w", err)
	}

	// Build constants map: model-config.ts object properties first,
	// then freebuff-models.ts exports (which may reference model-config props).
	constants := make(map[string]string)
	constants = parseConstants(modelConfigSrc, constants)
	constants = parseConstants(freebuffModelsSrc, constants)

	all := parseAllFreeModels(freeAgentsSrc, constants)
	if len(all) == 0 {
		return fmt.Errorf("no free agents found in source")
	}

	modelToAgent, allModels := buildModelMapping(all)

	r.mu.Lock()
	r.agentModels = all
	r.modelToAgent = modelToAgent
	r.allModels = allModels
	r.lastOK = time.Now()
	r.mu.Unlock()

	r.logger.Printf("model registry: updated %d agents, %d models: %v", len(all), len(allModels), allModels)
	return nil
}

func (r *ModelRegistry) loadFallback() {
	modelToAgent, allModels := buildModelMapping(hardcodedFallback)

	r.mu.Lock()
	r.agentModels = hardcodedFallback
	r.modelToAgent = modelToAgent
	r.allModels = allModels
	r.mu.Unlock()

	r.logger.Printf("model registry: loaded fallback models: %v", allModels)
}

// fetchSource fetches a raw source file from a URL and returns the body as a string.
func fetchSource(ctx context.Context, client *http.Client, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Accept", "text/plain")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("unexpected status %d from %s", resp.StatusCode, url)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response from %s: %w", url, err)
	}

	return string(body), nil
}

// parseConstants extracts constant→value mappings from TypeScript source files.
// Handles two patterns:
//
//	export const FREEBUFF_X_MODEL_ID = 'minimax/minimax-m3'        → direct string
//	export const FREEBUFF_KIMI_MODEL_ID = moonshotModels.kimiK26   → resolves object ref
//
// Object properties (e.g. in model-config.ts) are stored with their parent object
// name as prefix: "minimaxModels.minimaxM3" → "minimax/minimax-m3".
func parseConstants(source string, existing map[string]string) map[string]string {
	if existing == nil {
		existing = make(map[string]string)
	}

	// Pattern 1: export const NAME = 'string-value'
	stringConstRe := regexp.MustCompile(`export\s+const\s+(\w+)\s*=\s*'([^']+)'`)
	for _, m := range stringConstRe.FindAllStringSubmatch(source, -1) {
		existing[m[1]] = m[2]
	}

	// Pattern 2: Object property definitions with tracked object context.
	// Matches:   export const objName = {  prop: 'value',  ...  }
	// Stores as: "objName.prop" → "value"
	// This handles model-config.ts exports like:
	//   export const minimaxModels = { minimaxM3: 'minimax/minimax-m3', ... }
	objDefRe := regexp.MustCompile(`export\s+const\s+(\w+)\s*=\s*\{([^}]+)\}`)
	for _, m := range objDefRe.FindAllStringSubmatch(source, -1) {
		objName := m[1]
		body := m[2]
		propRe := regexp.MustCompile(`(\w+)\s*:\s*'([^']+)'`)
		for _, pm := range propRe.FindAllStringSubmatch(body, -1) {
			key := objName + "." + pm[1]
			existing[key] = pm[2]
		}
	}

	// Pattern 3: export const NAME = objectName.property
	// Resolves references like moonshotModels.kimiK26 using already-known values.
	objConstRe := regexp.MustCompile(`export\s+const\s+(\w+)\s*=\s*(\w+)\.(\w+)`)
	for _, m := range objConstRe.FindAllStringSubmatch(source, -1) {
		lookupKey := m[2] + "." + m[3]
		if val, ok := existing[lookupKey]; ok {
			existing[m[1]] = val
		}
	}

	return existing
}

// resolveModelRef resolves a single model entry from a Set([...]) list.
// If it's a quoted string literal, it's returned as-is.
// If it's an identifier like FREEBUFF_MINIMAX_MODEL_ID, it's looked up in constants.
func resolveModelRef(ref string, constants map[string]string) string {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return ""
	}
	// Quoted string literal
	if len(ref) >= 2 && ref[0] == '\'' && ref[len(ref)-1] == '\'' {
		return ref[1 : len(ref)-1]
	}
	// Constant reference — look it up
	if val, ok := constants[ref]; ok {
		return val
	}
	return ""
}

// parseAllFreeModels extracts ALL agent→models mappings from the free-agents.ts source.
// constants maps TypeScript constant names to their string values, enabling
// resolution of identifiers like FREEBUFF_MINIMAX_MODEL_ID.
func parseAllFreeModels(source string, constants map[string]string) map[string][]string {
	blockPattern := regexp.MustCompile(`'([^']+)':\s*new\s+Set\(\[([^\]]*)\]\)`)

	// Match either 'quoted-string' or bare identifiers (FREEBUFF_*, etc.)
	entryPattern := regexp.MustCompile(`'([^']+)'|([A-Z_][A-Z0-9_]*)`)

	result := make(map[string][]string)
	for _, match := range blockPattern.FindAllStringSubmatch(source, -1) {
		agentID := match[1]
		modelsStr := match[2]

		var models []string
		for _, entry := range entryPattern.FindAllStringSubmatch(modelsStr, -1) {
			var model string
			if entry[1] != "" {
				// Quoted string literal
				model = entry[1]
			} else if entry[2] != "" {
				// Constant reference — resolve via constants map
				model = resolveModelRef(entry[2], constants)
			}
			model = strings.TrimSpace(model)
			if model != "" {
				models = append(models, model)
			}
		}
		if len(models) > 0 {
			result[agentID] = models
		}
	}
	return result
}

// buildModelMapping creates the model→agent reverse mapping and deduplicated model list.
// When a model appears in multiple agents, one is chosen at random.
func buildModelMapping(agentModels map[string][]string) (map[string]string, []string) {
	modelAgents := make(map[string][]string)
	for agentID, models := range agentModels {
		for _, model := range models {
			modelAgents[model] = append(modelAgents[model], agentID)
		}
	}

	modelToAgent := make(map[string]string, len(modelAgents))
	allModels := make([]string, 0, len(modelAgents))
	for model, agents := range modelAgents {
		modelToAgent[model] = agents[rand.Intn(len(agents))]
		allModels = append(allModels, model)
	}
	sort.Strings(allModels)
	return modelToAgent, allModels
}
