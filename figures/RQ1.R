# install.packages(c("tidyverse","scales","ggtext")) # if needed
library(tidyverse)
library(scales)

# Palette and theme (match the shown style)
col_blue  <- "#2E86C1"  # bars 1 (blue)
col_orng  <- "#E67E22"  # bars 2 (orange)
col_green <- "#27AE60"  # bars 3 (green)
bg_grey   <- "#F5F6F7"
grid_grey <- "#E5E7E9"

theme_fig <- theme_minimal(base_size = 14) +
  theme(
    plot.title      = element_text(face = "bold", size = 20, hjust=0.5),
    plot.subtitle   = element_text(size = 13, color = "grey20", hjust=0.5),
    plot.caption    = element_text(size = 10, color = "grey35"),
    legend.position = "bottom",
    legend.title    = element_text(size = 12),
    panel.grid.major.y = element_line(color = grid_grey),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = bg_grey, color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    strip.text = element_text(face = "bold", size = 16),
    axis.title.x = element_text(margin = margin(t = 8)),
    axis.title.y = element_text(margin = margin(r = 8))
  )

# Data
df <- tribble(
  ~Provider,   ~Model,        ~Score,
  "Qwen CLI",   "Llama3",       0.806,
  "Qwen CLI",   "GPT5-mini",    0.803,
  "Qwen CLI",   "GPT4o-mini",   0.775,
  "Qwen CLI",   "GPT4o",        0.792,
  "Qwen CLI",   "GPT4.1-mini",  0.781,
  "Gemini CLI", "Llama3",       0.958,
  "Gemini CLI", "GPT5-mini",    0.964,
  "Gemini CLI", "GPT4o-mini",   0.955,
  "Gemini CLI", "GPT4o",        0.962,
  "Gemini CLI", "GPT4.1-mini",  0.956
) %>%
  mutate(
    Provider = factor(Provider, levels = c("Qwen CLI","Gemini CLI")),
    Model = factor(Model, levels = c("Llama3","GPT5-mini","GPT4o-mini","GPT4o","GPT4.1-mini"))
  )

# 1) Grouped bar (colors: blue for Qwen, orange for Gemini), labels on bars
p_grouped <- ggplot(df, aes(Model, Score, fill = Provider)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.68) +
  geom_text(aes(label = percent(Score, accuracy = 0.1)),
            position = position_dodge(width = 0.8), vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = c("Qwen CLI" = col_blue, "Gemini CLI" = col_orng)) +
  scale_y_continuous(labels = percent, limits = c(0,1)) +
  labs(
    title = "ASR by Model and Agent",
    x = NULL, y = "Percentage (%)",
    fill = "Agent",
    caption = "471 attack instances instances"
  ) +
  theme_fig 
print(p_grouped)
ggsave("/Users/ryan0079/Documents/GitHub/AgentJailbreak/figures/RQ1.png", p_grouped, width = 10, height = 5, dpi = 200)

# Side-by-side panels (one for Qwen CLI, one for Gemini CLI)
p_side <- ggplot(df, aes(Model, Score, fill = Provider)) +
  geom_col(width = 0.68, show.legend = FALSE) +
  geom_text(aes(label = percent(Score, accuracy = 0.1)),
            vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = c("Qwen CLI" = col_blue, "Gemini CLI" = col_orng)) +
  scale_y_continuous(labels = percent, limits = c(0,1)) +
  labs(
    title = "ASR by Model and Agent",
    x = NULL, y = "Percentage (%)",
    caption = "471 attack instances instances"
  ) +
  facet_wrap(~ Provider, nrow = 1) +
  theme_fig

print(p_side)
ggsave("/Users/ryan0079/Documents/GitHub/AgentJailbreak/figures/RQ1.png",
       p_side, width = 10, height = 5, dpi = 200)




# 2) Faceted bars (per provider), use provider color + green outline hint
p_facets <- ggplot(df, aes(Model, Score, fill = Provider)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_text(aes(label = percent(Score, accuracy = 0.1)),
            vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = c("Qwen CLI" = col_blue, "Gemini CLI" = col_orng)) +
  scale_y_continuous(labels = percent, limits = c(0,1)) +
  facet_wrap(~ Provider, ncol = 1) +
  labs(
    title = "ASR per Provider",
    subtitle = "Same styling and colors as reference",
    x = NULL, y = "Percentage (%)"
  ) +
  theme_fig
print(p_facets)

# 3) Heatmap (tile) with labels; keep palette but use blue→green gradient for fill
p_heat <- ggplot(df, aes(Model, Provider, fill = Score)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = percent(Score, accuracy = 0.1)), color = "white", size = 3.6, fontface = "bold") +
  scale_fill_gradient(low = col_blue, high = col_green, labels = percent, limits = c(0,1)) +
  labs(
    title = "ASR Heatmap",
    subtitle = "Higher is better",
    x = NULL, y = NULL, fill = "ASR"
  ) +
  theme_fig
print(p_heat)

# Optional saves
# ggsave("asr_grouped.png", p_grouped, width = 10, height = 5, dpi = 200)
# ggsave("asr_facets.png",  p_facets,  width = 9,  height = 6, dpi = 200)
# ggsave("asr_heatmap.png", p_heat,    width = 9,  height = 3.8, dpi = 200)