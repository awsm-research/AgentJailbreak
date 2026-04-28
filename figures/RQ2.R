library(tidyverse)
library(scales)

# Original counts
df <- tribble(
  ~Agent,  ~Category,                  ~Count,
  "Gemini","AcknowledgedButDidNotRun",  50,
  "Gemini","IrrelevantToTask",           3,
  "Gemini","NoAcknowledgement",         24,
  "Gemini","RecognizedAttack",          15,
  "Qwen",  "AcknowledgedButDidNotRun",  91,
  "Qwen",  "IrrelevantToTask",          45,
  "Qwen",  "NoAcknowledgement",        274,
  "Qwen",  "RecognizedAttack",          56,
)

# Remove "Other" and recompute totals/props
df_no_other <- df %>% filter(Category != "Other")

# Remove "Other" and recompute totals/props
plot_df <- df %>%
  filter(Category != "Other") %>%
  mutate(Agent = factor(Agent, levels = c("Gemini","Qwen"))) %>%
  group_by(Agent) %>%
  mutate(
    Total = sum(Count),
    Prop  = Count / Total,
    Label = if_else(Prop >= 0.03, paste0(Category, " (", percent(Prop, 0.1), ")"), "")
  ) %>%
  ungroup()

# Monochrome blues (light -> dark); keys are ORIGINAL category names
pal <- c(
  "AcknowledgedButDidNotRun" = "#CFE8F3",
  "NoAcknowledgement"        = "#91C3E6",
  "RecognizedAttack"         = "#4F97C7",
  "IrrelevantToTask"         = "#1F6AA5"
)

# Legend with fuller descriptions, but slices keep original names
legend_labels <- c(
  "AcknowledgedButDidNotRun" = "Acknowledged skills file but did not attempt to run",
  "NoAcknowledgement"        = "Did not acknowledge skills file",
  "RecognizedAttack"         = "Recognized as attack + explicit refusal",
  "IrrelevantToTask"         = "Refused as irrelevant to the task"
)

theme_clean <- theme_void(base_size = 13) +
  theme(
    strip.text = element_text(face = "bold", size = 16),
    legend.position = "bottom",
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5)
  )

lab_df <- plot_df %>%
  group_by(Agent) %>% arrange(desc(Category)) %>%
  mutate(
    pos   = cumsum(Prop) - Prop/2,   # mid-slice (on y-scale for polar)
    Label = if_else(Prop >= 0.03, paste0(Category, " (", percent(Prop, 0.1), ")"), NA_character_)
  ) %>%
  filter(!is.na(Label)) %>%
  ungroup()


# Full circular pies
p <- ggplot(plot_df, aes(x = "", y = Prop, fill = Category)) +
  geom_col(color = "white", width = 1, linewidth = 0.6) +
  coord_polar(theta = "y") +
  facet_wrap(~ Agent, nrow = 1) +
  scale_fill_manual(
    values = pal,
    breaks = names(legend_labels),
    labels = legend_labels,
    name = "Failed Exploitation Cause"
  ) +
  guides(fill = guide_legend(ncol = 1)) +
  theme(
    legend.position = "bottom",
    legend.direction = "vertical",
    legend.box = "vertical"
  ) +
  # percent-only labels inside slices
  geom_text(
    aes(label = scales::percent(Prop, accuracy = 0.1)),
    position = position_stack(vjust = 0.5),
    color = "black", size = 3.6, fontface = "bold"
  ) +
  labs(title = "Failure Categories by Agent") +
  theme_clean +
  theme(aspect.ratio = 1)

print(p)

ggsave("/Users/ryan0079/Documents/GitHub/AgentJailbreak/figures/RQ2.png", p, width = 8, height = 5, dpi = 200)