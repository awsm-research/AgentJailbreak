library(tidyverse); library(scales)

tactics <- tribble(
  ~Tactic,              ~Succ, ~Tot,
  "defense-evasion",     1093, 1192,
  "initial-access",       192,  210,
  "discovery",            714,  801,
  "privilege-escalation", 693,  779,
  "execution",            393,  450,
  "credential-access",    454,  519,
  "persistence",           67,   80,
  "impact",               182,  220,
  "collection",           132,  160,
  "command-and-control",  151,  190,
  "exfiltration",          81,  110
) %>%
  mutate(ER = Succ/Tot,
         Tactic = fct_reorder(Tactic, ER))

p_tactic_overall <- ggplot(tactics, aes(ER, Tactic)) +
  geom_col(fill = "#4F97C7") +
  geom_text(aes(label = percent(ER, 0.1)), hjust = -0.05, size = 3.6, fontface = "bold") +
  scale_x_continuous(labels = percent, limits = c(0, 1.05)) +
  labs(title = "Exploitability by ATT&CK Tactic (overall)",
       x = "Exploitability Rate", y = NULL,
       caption = "ER = successes / evaluated runs") +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank())

print(p_tactic_overall)

# ggsave("/Users/ryan0079/Documents/GitHub/AgentJailbreak/figures/RQ3_tactic_overall.png", p_tactic_overall, width=8, height=5, dpi=200)


df <- read_csv("/Users/ryan0079/Documents/GitHub/AgentJailbreak/assessed_results/final/exploit_by_tactic.csv", show_col_types = FALSE)
# expected cols: tactic, agent, success, total, er
df <- df %>%
  mutate(agent = str_to_title(agent),
         tactic = fct_reorder(tactic, er)) %>%
  filter(agent %in% c("Gemini","Qwen"))  # keep two agents

p_tactic_agent <- ggplot(df, aes(er, tactic, fill = agent)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "white") +
  geom_text(aes(label = percent(er, 0.1)),
            position = position_dodge(width = 0.8), hjust = -0.05, size = 3.2) +
  scale_fill_manual(values = c("Gemini"="#4F97C7","Qwen"="#1F6AA5"), name = "Agent") +
  scale_x_continuous(labels = percent, limits = c(0, 1.05)) +
  labs(title = "Exploitability by Tactic and Agent",
       x = "Exploitability Rate", y = NULL) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank())

print(p_tactic_agent)

# ggsave("/Users/ryan0079/Documents/GitHub/AgentJailbreak/figures/RQ3_tactic_by_agent.png", p_tactic_agent, width=9, height=5, dpi=200)


tech <- read_csv("/Users/ryan0079/Documents/GitHub/AgentJailbreak/assessed_results/final/exploit_by_tech.csv", show_col_types = FALSE)
tech <- tech %>% filter(agent %in% c("gemini","qwen")) %>%
  mutate(agent=str_to_title(agent)) %>%
  group_by(agent) %>% slice_max(order_by = er, n = 15, with_ties = FALSE) %>%
  ungroup() %>% mutate(technique = fct_reorder(technique, er))

p_tech_agent <- ggplot(tech, aes(er, technique, fill = agent)) +
  geom_col(position = position_dodge(width=0.8), width=0.7) +
  scale_x_continuous(labels = percent, limits = c(0,1.05)) +
  scale_fill_manual(values = c("Gemini"="#4F97C7","Qwen"="#1F6AA5"), name="Agent") +
  labs(title="Top Techniques by Exploitability (per Agent)",
       x="Exploitability Rate", y=NULL) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank())

print(p_tech_agent)
# ggsave("/Users/ryan0079/Documents/GitHub/AgentJailbreak/figures/RQ3_tech_by_agent_top15.png", p_tech_agent, width=10, height=7, dpi=200)