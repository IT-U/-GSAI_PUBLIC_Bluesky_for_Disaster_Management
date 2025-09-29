# install scripts
install.packages("sysfonts")
install.packages("showtext")
install.packages("sf")
install.packages("arrow")
install.packages("knitr")
install.packages("kableExtra")
install.packages("patchwork")
install.packages("geoarrow")

# imports
library(geoarrow)     # for geoparquet files
library(arrow)        # for reading Parquet
library(sf)           # for spatial data
library(dplyr)        # data manipulation
library(lubridate)    # date parsing
library(tidyr)        # for completeness
library(ggplot2)      # plotting
library(patchwork)    # arranging multiple ggplots
library(knitr)        # tables
library(kableExtra)   # LaTeX styling for tables
library(sysfonts)     # for custom fonts
library(showtext)     # for custom fonts
# library(ggimage)      # images in plots
library(RColorBrewer) # colors

# Add Google font
font_add_google(name = "Barlow", family = "barlow")
showtext_auto()

# Define paths
PROJECT_ROOT <- "C:/Users/DavidHanny/OneDrive - IT U interdisciplinary transformation university austria/Documents/projects/papers/2025_GSAI_RES_LLM_Contextual_Predictions"
# PROJECT_ROOT <- "/mnt/c/Users/DavidHanny/OneDrive - IT U interdisciplinary transformation university austria/Documents/projects/papers/2025_GSAI_RES_LLM_Contextual_Predictions"
setwd(PROJECT_ROOT)
getwd()
DATA_PATH    <- file.path(PROJECT_ROOT, "data")
FIGURE_PATH  <- file.path(PROJECT_ROOT, "figures", "bsky_paper")


######################
# 1. Data preparation
######################

# Read Bluesky GeoParquet datasets
bsky_socal <- read_parquet(file.path(DATA_PATH, "processed", "2025_socal_wildfires", "bsky_socal_gdf_esda.parquet"))
bsky_europe <- read_parquet(file.path(DATA_PATH, "processed", "2024_central_europe_floods", "bsky_central_europe_gdf_esda.parquet"))

# Convert datasets to sf
bsky_europe$geometry <- st_as_sfc(bsky_europe$geometry)
bsky_europe <- st_as_sf(bsky_europe)
bsky_socal$geometry <- st_as_sfc(bsky_socal$geometry)
bsky_socal <- st_as_sf(bsky_socal)

# Compute helper flags on both datasets
for (df_name in c("bsky_socal","bsky_europe")) {
  df <- get(df_name)
  df <- df %>%
    mutate(
      contains_external_url = sapply(urls, function(x) any(grepl("http", x))),
      contains_image        = sapply(image_thumbnails, length) > 0
    )
  assign(df_name, df)
}

######################
# 2. General plots
######################

# -----------------------------------------------------------------------------
# Language distribution plots
# -----------------------------------------------------------------------------

make_lang_plot <- function(gdf, title_text) {
  df <- gdf %>%
    mutate(language = if_else(is.na(language), "unk", language)) %>%
    count(language, name="abs_count") %>%
    arrange(desc(abs_count))
  
  top9 <- df %>% slice_head(n=5)  # initially was top 10, but maybe the top 5 suffices
  other_count <- sum(df$abs_count) - sum(top9$abs_count)
  df2 <- bind_rows(
    top9,
    tibble(language = "other", abs_count = other_count)
  ) %>%
    mutate(
      rel_count = abs_count / sum(abs_count) * 100,
      language = factor(language, levels = rev(language))
    )
  
  ggplot(df2, aes(x=rel_count, y=language)) +
    geom_col(width=0.7) +
    geom_text(aes(label=sprintf("%.1f%%", rel_count)),
              hjust = -0.1, size = 3) +
    scale_x_continuous(expand = expansion(mult = c(0,0.1))) +
    labs(
      title = title_text,
      x     = "Fraction of posts (%)",
      y     = "Language"
    ) +
    theme_minimal() +
    theme(
      panel.grid.major.y = element_blank(),
      panel.grid.minor   = element_blank(),
      plot.title         = element_text(face="bold", size=14),
      axis.title.y       = element_blank(),
      axis.ticks.y       = element_blank(),
      text               = element_text(family="barlow", size=14),
      axis.text          = element_text(size = 12),  # Makes axis labels larger
      axis.title.x       = element_text(size = 12)   # Makes x-axis title larger
    )
}

p1 <- make_lang_plot(bsky_europe, "2024 Central Europe floods")
p2 <- make_lang_plot(bsky_socal,  "2025 Southern California wildfires")

# Combine and save
(p1 + p2) +
  plot_annotation(
    title = "Relative distribution of languages across posts",
    theme = theme(
      plot.title = element_text(face = "bold", size = 16)
    )
  )

ggsave(
  filename = file.path(FIGURE_PATH, "ggplot", "language_distribution.pdf"),
  width    = 10, height = 5, units = "in", dpi = 300
)


######################
# 3. Central Europe plots
######################

# -----------------------------------------------------------------------------
# Disaster-relatedness
# -----------------------------------------------------------------------------

europe_polygon <- st_as_sfc(
  "POLYGON (
     (1.40625 35.746512,
      32.695313 35.746512,
      32.695313 57.04073,
      1.40625 57.04073,
      1.40625 35.746512)
  )",
  crs = st_crs(bsky_europe)
)

europe_daily_disaster_df <- bsky_europe %>%
  filter(
    st_intersects(geometry, europe_polygon, sparse = FALSE)[,1]
  ) %>% 
  st_drop_geometry() %>%
  mutate(date = as_date(createdAt)) %>%
  group_by(date) %>%
  summarize(total = n(),
            disaster = sum(disaster_related, na.rm=TRUE)) %>%
  ungroup() %>%
  complete(date = seq(min(date), max(date), by = "day"),
           fill = list(total = 0, disaster = 0)) %>%
  mutate(prop_disaster = if_else(total>0, 100*disaster/total, 0))

# For the plot itself: define the two annotation dates
annots <- tibble(
  date = as_date(c("2024-09-14","2024-09-23")),
  label = c("Heavy rain\nand flooding","Aftermath\nand recovery"),
  icon  = c("figures/bsky_paper/diagrams/icons/rainy.png",
            "figures/bsky_paper/diagrams/icons/flood.png")
)

ggplot(europe_daily_disaster_df, aes(x = date, y = prop_disaster)) +
  # shaded event windows (if desired; remove if not)
  annotate("rect", xmin = as_date("2024-09-12"), xmax = as_date("2024-09-16"),
           ymin = -Inf, ymax = Inf, fill = "#F8CECC", alpha = 0.3) +
  annotate("rect", xmin = as_date("2024-09-16"), xmax = as_date("2024-09-30"),
           ymin = -Inf, ymax = Inf, fill = "#FFF2CC", alpha = 0.3) +
  geom_line(size=1.2) +
  geom_point(size=3) +
  scale_y_continuous(name = "Proportion of disaster-related posts (%)",
                     limits = c(0, max(europe_daily_disaster_df$prop_disaster)*1.1)) +
  scale_x_date(date_labels = "%b %d %Y") +
  labs(
    title = "Daily proportion of disaster-related posts\n(2024 Central Europe floods)",
    x = "Date"
  ) +
  # text labels describing the situation
  geom_text(data = annots,
            aes(x = date, y = max(europe_daily_disaster_df$prop_disaster) * 0.25, label = label),
            hjust = 0.5, vjust = 0, size = 4.5,
            family = "barlow", face = "bold", lineheight = 0.95) +
  theme_minimal() +
  theme(
    text               = element_text(family="barlow", size=14),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.border       = element_blank(),
    plot.title         = element_text(face = "bold", size = 16),
    panel.grid.minor.y = element_blank(),
    axis.text          = element_text(size = 13),  # Makes axis labels larger
    axis.title.x       = element_text(size = 14),  # Makes x-axis title larger
    axis.title.y       = element_text(size = 14)   # Makes x-axis title larger
  )

ggsave(
  filename = file.path(FIGURE_PATH, "ggplot", "daily_disaster_related_posts_central_europe.pdf"),
  width    = 11, height = 5, units = "in", dpi = 300
)


# -----------------------------------------------------------------------------
# Emotions
# -----------------------------------------------------------------------------

emotion_cols <- c("anger", "fear", "joy", "sadness", "no_emotion")
label_map <- c(
  anger      = "Anger",
  fear       = "Fear",
  joy        = "Joy",
  sadness    = "Sadness",
  no_emotion = "No Emotion"
)

europe_emotion_df <- bsky_europe %>%
  filter(st_intersects(geometry, europe_polygon, sparse = FALSE)[,1]) %>%
  st_drop_geometry() %>%
  mutate(date = as_date(createdAt)) %>%
  group_by(date) %>%
  summarise(
    total      = n(),
    anger      = sum(anger,      na.rm = TRUE),
    fear       = sum(fear,       na.rm = TRUE),
    joy        = sum(joy,        na.rm = TRUE),
    sadness    = sum(sadness,    na.rm = TRUE),
    no_emotion = sum(no_emotion, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  complete(
    date = seq(min(date), max(date), by = "day"),
    fill = list(
      total      = 0,
      anger      = 0,
      fear       = 0,
      joy        = 0,
      sadness    = 0,
      no_emotion = 0
    )
  ) %>% pivot_longer(
    cols      = c(anger, fear, joy, sadness, no_emotion),
    names_to  = "emotion",
    values_to = "count"
  ) %>%
  mutate(
    percentage    = if_else(total > 0, 100 * count / total, 0),
    emotion_label = label_map[emotion]
  )


# Annotation data with icons
annots <- tibble(
  date  = as_date(c("2024-09-14", "2024-09-23")),
  label = c("Heavy rain\nand flooding", "Aftermath\nand recovery"),
  icon  = c(
    "figures/bsky_paper/diagrams/icons/rainy.png",
    "figures/bsky_paper/diagrams/icons/flood.png"
  )
)
max_pct <- max(europe_emotion_df$percentage, na.rm = TRUE)

# Main plot
ggplot(europe_emotion_df, aes(x = date, y = percentage, color = emotion_label, shape = emotion_label)) +
  # shaded event windows (if desired; remove if not)
  annotate("rect", xmin = as_date("2024-09-12"), xmax = as_date("2024-09-16"),
           ymin = -Inf, ymax = Inf, fill = "#F8CECC", alpha = 0.3) +
  annotate("rect", xmin = as_date("2024-09-16"), xmax = as_date("2024-09-30"),
           ymin = -Inf, ymax = Inf, fill = "#FFF2CC", alpha = 0.3) +
  # main data
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  # scales
  scale_color_brewer(palette = "Set1") +
  # scale_shape_manual(values = c(16,6,17,18,8)) +
  scale_x_date(date_labels = "%b %d, %Y") +
  scale_y_continuous(
    name   = "Proportion of posts (%)",
    limits = c(0, max_pct * 1.1)
  ) +
  # text labels describing the situation
  geom_text(data = annots, inherit.aes = FALSE,
            aes(x = date, y = max_pct * 1, label = label),
            hjust = 0.5, vjust = 0, size = 4.5,
            family = "barlow", lineheight = 0.95) +
  # titles & legend
  labs(
    title = "Daily proportion of emotions across Bluesky posts\n(2024 Central Europe floods)",
    x     = "Date",
    color = "Emotion",
    shape = "Emotion"
  ) +
  # design
  theme_minimal() +
  theme(
    text               = element_text(family="barlow", size=14),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.border       = element_blank(),
    plot.title         = element_text(face = "bold", size = 16),
    panel.grid.minor.y = element_blank(),
    axis.text          = element_text(size = 13),  # Makes axis labels larger
    axis.title.x       = element_text(size = 14),  # Makes x-axis title larger
    axis.title.y       = element_text(size = 14)   # Makes x-axis title larger
  )

ggsave(
  filename = file.path(FIGURE_PATH, "ggplot", "daily_emotions_central_europe.pdf"),
  width    = 11, height = 5, units = "in", dpi = 300
)

# -----------------------------------------------------------------------------
# Histogram of posts
# -----------------------------------------------------------------------------
histogram_data_europe <- st_read(
  file.path(DATA_PATH, "processed", "bsky_spatial", 
            "bluesky_relevant_posts_central_europe_h3_level_4_intersects_flood.gpkg")
)

# Compute the maximum value of 'count' (safely handling NA)
max_count <- max(histogram_data_europe$count, na.rm = TRUE)

# Define fixed bin edges, then adjust to include the actual max
bin_edges <- c(1, 5, 10, 20, 50, 100, 200, 500)
if (max_count > bin_edges[length(bin_edges)]) {
  bin_edges <- c(bin_edges, max_count + 1)
} else {
  # Keep only edges ≤ max_count, then append max_count + 1
  bin_edges <- bin_edges[bin_edges <= max_count]
  bin_edges <- c(bin_edges, max_count + 1)
}

# Define matching labels for the bins
all_labels <- c("1–4", "5–9", "10–19", "20–49", "50–99", "100–199", "200–499", "500+")
labels <- all_labels[seq_len(length(bin_edges) - 1)]

# Create a new histogram column 'count_bin' using cut()
histogram_data_europe <- histogram_data_europe %>%
  mutate(
    count_bin = cut(
      count,
      breaks = bin_edges,
      labels = labels,
      right = FALSE,
      include.lowest = TRUE
    )
  )

# Summarize counts by (count_bin, intersects_flood)
# Drop geometry before grouping
grouped <- histogram_data_europe %>%
  st_drop_geometry() %>%
  group_by(count_bin, intersects_flood) %>%
  summarise(n = n(), .groups = "drop")

# Step 7: Pivot to wide format so we have separate columns for Flood vs. No Flood
pivot <- grouped %>%
  pivot_wider(
    names_from = intersects_flood,
    values_from = n,
    values_fill = list(n = 0)
  ) %>%
  rename(
    `Flood`   = `TRUE`,
    `No Flood` = `FALSE`
  ) %>%
  arrange(factor(count_bin, levels = labels))

# (Optional) If you want a long-format data frame for ggplot directly:
pivot_long <- pivot %>%
  pivot_longer(
    cols = c(`No Flood`, Flood),
    names_to = "intersects_flood",
    values_to = "n"
  ) %>%
  mutate(
    # Ensure the factor levels match the intended order
    count_bin = factor(count_bin, levels = labels),
    intersects_flood = factor(intersects_flood, levels = c("No Flood", "Flood"))
  ) %>% filter(!is.na(count_bin))

# Plot a stacked bar chart with ggplot2
ggplot(pivot_long, aes(x = count_bin, y = n, fill = intersects_flood)) +
  geom_bar(stat = "identity") +
  labs(
    x = "Range of values (disaster-related posts within cell)",
    y = "Number of observed cells",
    title = "Number of disaster-related posts within H3 grid cells\n(2024 Central Europe floods)",
  ) +
  scale_fill_brewer(
    palette = "Set1",
    name   = "Intersects flooded area?",
    labels = c(
      "No Flood" = "No flood",
      "Flood"    = "Flood"
    )
  ) +
  theme_minimal(base_size = 14) +
  theme(
    text               = element_text(family="barlow", size=14),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.border       = element_blank(),
    plot.title         = element_text(face = "bold", size = 16),
    axis.text          = element_text(size = 13),  # Makes axis labels larger
    axis.title.x       = element_text(size = 14),  # Makes x-axis title larger
    axis.title.y       = element_text(size = 14)   # Makes x-axis title larger
    # panel.grid.minor.y = element_blank(),
  ) -> histogram_plot_europe
histogram_plot_europe

ggsave(
  histogram_plot_europe,
  filename = file.path(FIGURE_PATH, "ggplot", "h3_grid_histogram_europe.pdf"),
  width    = 11, height = 5, units = "in", dpi = 300
)

######################
# 3. Southern California plots
######################

# -----------------------------------------------------------------------------
# Disaster-relatedness
# -----------------------------------------------------------------------------
# define region of California
active_wildfire_delinations <- st_as_sfc(
  'POLYGON((-124.4009 41.9983,-123.6237 42.0024,-123.1526 42.0126,-122.0073 42.0075,-121.2369 41.9962,-119.9982 41.9983,-120.0037 39.0021,-117.9575 37.5555,-116.3699 36.3594,-114.6368 35.0075,-114.6382 34.9659,-114.6286 34.9107,-114.6382 34.8758,-114.5970 34.8454,-114.5682 34.7890,-114.4968 34.7269,-114.4501 34.6648,-114.4597 34.6581,-114.4322 34.5869,-114.3787 34.5235,-114.3869 34.4601,-114.3361 34.4500,-114.3031 34.4375,-114.2674 34.4024,-114.1864 34.3559,-114.1383 34.3049,-114.1315 34.2561,-114.1651 34.2595,-114.2249 34.2044,-114.2221 34.1914,-114.2908 34.1720,-114.3237 34.1368,-114.3622 34.1186,-114.4089 34.1118,-114.4363 34.0856,-114.4336 34.0276,-114.4652 34.0117,-114.5119 33.9582,-114.5366 33.9308,-114.5091 33.9058,-114.5256 33.8613,-114.5215 33.8248,-114.5050 33.7597,-114.4940 33.7083,-114.5284 33.6832,-114.5242 33.6363,-114.5393 33.5895,-114.5242 33.5528,-114.5586 33.5311,-114.5778 33.5070,-114.6245 33.4418,-114.6506 33.4142,-114.7055 33.4039,-114.6973 33.3546,-114.7302 33.3041,-114.7206 33.2858,-114.6808 33.2754,-114.6698 33.2582,-114.6904 33.2467,-114.6794 33.1720,-114.7083 33.0904,-114.6918 33.0858,-114.6629 33.0328,-114.6451 33.0501,-114.6286 33.0305,-114.5888 33.0282,-114.5750 33.0351,-114.5174 33.0328,-114.4913 32.9718,-114.4775 32.9764,-114.4844 32.9372,-114.4679 32.8427,-114.5091 32.8161,-114.5311 32.7850,-114.5284 32.7573,-114.5641 32.7503,-114.6162 32.7353,-114.6986 32.7480,-114.7220 32.7191,-115.1944 32.6868,-117.3395 32.5121,-117.4823 32.7838,-117.5977 33.0501,-117.6814 33.2341,-118.0591 33.4578,-118.6290 33.5403,-118.7073 33.7928,-119.3706 33.9582,-120.0050 34.1925,-120.7164 34.2561,-120.9128 34.5360,-120.8427 34.9749,-121.1325 35.2131,-121.3220 35.5255,-121.8013 35.9691,-122.1446 36.2808,-122.1721 36.7268,-122.6871 37.2227,-122.8903 37.7783,-123.2378 37.8965,-123.3202 38.3449,-123.8338 38.7423,-123.9793 38.9946,-124.0329 39.3088,-124.0823 39.7642,-124.5314 40.1663,-124.6509 40.4658,-124.3144 41.0110,-124.3419 41.2386,-124.4545 41.7170,-124.4009 41.9983,-124.4009 41.9983))',
crs=st_crs(bsky_socal)) %>% 
  st_make_valid()

# Filter posts to those that intersect the polygon
bsky_socal_regional <- bsky_socal %>%
  filter(st_intersects(geometry, active_wildfire_delinations, sparse = FALSE)[,1])

# Compute daily totals & disaster counts
daily_disaster_posts_socal <- bsky_socal_regional %>%
  mutate(date = as_date(createdAt)) %>%
  group_by(date) %>%
  summarise(
    total    = n(),
    disaster = sum(disaster_related, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  complete(
    date = seq(min(date), max(date), by = "day"),
    fill = list(total = 0, disaster = 0)
  ) %>%
  mutate(prop_disaster = if_else(total>0, 100*disaster/total, 0))

# read in fire delinations
fire_delineations <- read_parquet(
  file.path(DATA_PATH, "raw", "2025_socal_wildfires", "nasa_firms", "modis_fire_delineations.parquet")
) %>%
  # Ensure ACQ_DATETIME is POSIXct
  mutate(ACQ_DATETIME = as_datetime(ACQ_DATETIME, tz = "UTC")) %>%
  # Drop geometry if present (we only need timestamp)
  st_drop_geometry()

# Compute daily fire counts
daily_fires <- fire_delineations %>%
  mutate(date = as_date(ACQ_DATETIME)) %>%
  group_by(date) %>%
  summarise(fire_count = n()) %>%
  ungroup()

# Ensure zero‐counts for Feb 1–14, 2025 if missing
full_date_sequence <- tibble(
  date = seq(
    from = min(daily_fires$date), 
    to   = as_date('2025-02-14'), 
    by   = "day"
  )
)
daily_fires <- full_date_sequence %>%
  left_join(daily_fires, by = "date") %>%
  replace_na(list(fire_count = 0))

# Compute max values for scaling
max_prop  <- max(daily_disaster_posts_socal$prop_disaster, na.rm = TRUE)
max_fires <- max(daily_fires$fire_count, na.rm = TRUE)

# Scale factor to map fire_count onto the same y‐range as prop_disaster
scaleFactor <- (max_prop * 1.1) / (max_fires * 1.0)

# Create a plotting‐ready data frame for fires, scaled
daily_fires <- daily_fires %>%
  mutate(fires_scaled = fire_count * scaleFactor)

# Annotation
annots <- tibble(
  date  = as_date(c("2025-01-09", "2025-01-22")),
  y_align = c(max_prop * 0, max_prop * 0.7),
  label = c("Wildfire\noutbreaks", "Containment phase"),
)

# main plot
ggplot() +
  # Shaded area windows
  annotate("rect", xmin = as_date("2025-01-07"), xmax = as_date("2025-01-11"),
           ymin = -Inf, ymax = Inf, fill = "#F8CECC", alpha = 0.3) +
  annotate("rect", xmin = as_date("2025-01-11"), xmax = as_date("2025-01-31"),
           ymin = -Inf, ymax = Inf, fill = "#FFF2CC", alpha = 0.3) +
  # Annotations
  geom_text(data = annots, inherit.aes = FALSE,
            aes(x = date, y = y_align, label = label),
            hjust = 0.5, vjust = 0, size = 4,
            family = "barlow", lineheight = 0.95) +
  # Proportion of disaster‐related posts (primary y‐axis)
  geom_line(data = daily_disaster_posts_socal, aes(x = date, y = prop_disaster), linewidth = 1.2) +
  geom_point(data = daily_disaster_posts_socal, aes(x = date, y = prop_disaster), size = 3) +
  # Active fire counts (secondary y‐axis, dashed line, square markers)
  geom_line(data = daily_fires, aes(x = date, y = fires_scaled), color="firebrick") +
  geom_point(data = daily_fires, aes(x = date, y = fires_scaled), shape=15, color="firebrick") +
  # Scales and axes
  scale_x_date(
    date_labels = "%b %d, %Y",
    expand      = expansion(mult = c(0.01, 0.01))
  ) +
  scale_y_continuous(
    name = "Proportion of disaster-related posts",
    # Show percent with no decimal places
    labels = function(x) paste0(round(x), "%"),
    limits = c(0, max_prop * 1.1),
    sec.axis = sec_axis(
      trans = ~ . / scaleFactor,
      name  = "Number of active fires"
    )
  ) +
  # Titles & theme
  labs(
    title = "Daily proportion of disaster-related posts & active fires\n(2025 Southern California wildfires)",
    x     = "Date"
  ) +
  theme_minimal() +
  theme(
    text               = element_text(family = "barlow", size = 14),
    plot.title         = element_text(face = "bold", size = 16, margin = margin(b = 10)),
    axis.title.y.left  = element_text(face = "bold", size = 14),
    axis.title.y.right = element_text(face = "bold", size = 12, color = "firebrick"),
    axis.text.y.right  = element_text(color = "firebrick"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.text          = element_text(size = 12),  # Makes axis labels larger
    axis.title.x       = element_text(size = 14),  # Makes x-axis title larger
    axis.title.y       = element_text(size = 14)   # Makes x-axis title larger
  )  -> p_socal_disaster_related
p_socal_disaster_related

ggsave(
  plot     = p_socal_disaster_related,
  filename = file.path(FIGURE_PATH, "ggplot", "daily_disaster_related_posts_socal_wildfires.pdf"),
  width    = 11,
  height   = 5,
  units    = "in",
  dpi      = 300
)


# -----------------------------------------------------------------------------
# Plot emotions and wildfires
# -----------------------------------------------------------------------------

# Define emotion columns and label map (same as before)
emotion_cols <- c("anger", "fear", "joy", "sadness", "no_emotion")
label_map <- c(
  anger      = "Anger",
  fear       = "Fear",
  joy        = "Joy",
  sadness    = "Sadness",
  no_emotion = "No Emotion"
)

# Compute daily emotion counts & proportions for SoCal region
daily_emotion_socal <- bsky_socal_regional %>%
  st_drop_geometry() %>%
  mutate(date = as_date(createdAt)) %>%
  group_by(date) %>%
  summarise(
    total      = n(),
    anger      = sum(anger,      na.rm = TRUE),
    fear       = sum(fear,       na.rm = TRUE),
    joy        = sum(joy,        na.rm = TRUE),
    sadness    = sum(sadness,    na.rm = TRUE),
    no_emotion = sum(no_emotion, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  complete(
    date = seq(min(date), max(date), by = "day"),
    fill = list(
      total      = 0,
      anger      = 0,
      fear       = 0,
      joy        = 0,
      sadness    = 0,
      no_emotion = 0
    )
  ) %>%
  pivot_longer(
    cols      = c(anger, fear, joy, sadness, no_emotion),
    names_to  = "emotion",
    values_to = "count"
  ) %>%
  mutate(
    percentage    = if_else(total > 0, 100 * count / total, 0),
    emotion_label = label_map[emotion]
  )

# Compute scaling factor for secondary axis
max_pct  <- max(daily_emotion_socal$percentage, na.rm = TRUE)
max_fires <- max(daily_fires$fire_count, na.rm = TRUE)
scaleFactor <- (max_pct * 1.1) / (max_fires * 1.0)

daily_fires <- daily_fires %>%
  mutate(fires_scaled = fire_count * scaleFactor)

# Annotation
annots <- tibble(
  date  = as_date(c("2025-01-09", "2025-01-22")),
  y_align = c(max_pct * 0.95, max_pct * 0.95),
  label = c("Wildfire\noutbreaks", "Containment phase"),
)

# Plot with ggplot2
ggplot() +
  # Shaded area windows
  annotate("rect", xmin = as_date("2025-01-07"), xmax = as_date("2025-01-11"),
           ymin = -Inf, ymax = Inf, fill = "#F8CECC", alpha = 0.3) +
  annotate("rect", xmin = as_date("2025-01-11"), xmax = as_date("2025-01-31"),
           ymin = -Inf, ymax = Inf, fill = "#FFF2CC", alpha = 0.3) +
  # Annotations
  geom_text(data = annots, inherit.aes = FALSE,
            aes(x = date, y = y_align, label = label),
            hjust = 0.5, vjust = 0.5, size = 3.5,
            family = "barlow", lineheight = 0.95) +
  # Emotion lines + points (primary y-axis)
  geom_line(data = daily_emotion_socal, aes(x = date, y = percentage, color = emotion_label), size = 1.2) +
  geom_point( data = daily_emotion_socal, aes(x = date, y = percentage, shape = emotion_label, color = emotion_label), size = 3) +
  # Active fire counts (secondary y-axis, dashed line, square markers)
  #geom_line(
  #  data = daily_fires, aes(x = date, y = fires_scaled), color="firebrick") +
  #geom_point(data = daily_fires, aes(x = date, y = fires_scaled), color = "firebrick", shape = 15) +
  # Color palette
  scale_color_brewer(palette = "Set1") +
  #scale_shape_manual(values = c(16,6,17,18,8)) +
  # Scales and axes
  scale_x_date(date_labels = "%b %d, %Y", expand = expansion(mult = c(0.01, 0.01))) +
  scale_y_continuous(
    name = "Proportion of posts (%)",
    labels = function(x) paste0(round(x), "%"),
    limits = c(0, max_pct * 1.1),
  ) +
  # Titles and theme
  labs(
    title = "Daily proportion of emotions & active wildfires\n(2025 Southern California wildfires)",
    x     = "Date",
    color = "Emotion",
    shape = "Emotion"
  ) +
  theme_minimal() +
  theme(
    text                = element_text(family = "barlow", size = 14),
    plot.title          = element_text(face = "bold", size = 16, margin = margin(b = 10)),
    axis.title.y.left   = element_text(face = "bold", size = 14),
    axis.title.y.right  = element_text(face = "bold", size = 12, color = "firebrick"),
    axis.text.y.right   = element_text(color = "firebrick"),
    panel.grid.major.x  = element_blank(),
    panel.grid.minor    = element_blank(),
    axis.text          = element_text(size = 12),  # Makes axis labels larger
    axis.title.x       = element_text(size = 14),  # Makes x-axis title larger
    axis.title.y       = element_text(size = 14)   # Makes x-axis title larger
  ) ->
  p_socal_emotions_fires
p_socal_emotions_fires

ggsave(
  plot     = p_socal_emotions_fires,
  filename = file.path(FIGURE_PATH, "ggplot", "daily_emotions_socal_wildfires.pdf"),
  width    = 11,
  height   = 5,
  units    = "in",
  dpi      = 300
)

# -----------------------------------------------------------------------------
# Histogram of posts
# -----------------------------------------------------------------------------
histogram_data_socal <- st_read(
  file.path(DATA_PATH, "processed", "bsky_spatial", 
            "bluesky_relevant_posts_california_h3_level_4_intersects_fire.gpkg")
)

# Compute the maximum value of 'count' (safely handling NA)
max_count <- max(histogram_data_socal$count, na.rm = TRUE)

# Define fixed bin edges, then adjust to include the actual max
bin_edges <- c(1, 10, 50, 100, 500, 1000, 10000)
if (max_count > bin_edges[length(bin_edges)]) {
  bin_edges <- c(bin_edges, max_count + 1)
} else {
  # Keep only edges ≤ max_count, then append max_count + 1
  bin_edges <- bin_edges[bin_edges <= max_count]
  bin_edges <- c(bin_edges, max_count + 1)
}

# Define matching labels for the bins
all_labels <- c("1–10", "11-50", "51-100", "101-500", "501-1,000", "1,001-10,000", "10,000+")
labels <- all_labels[seq_len(length(bin_edges) - 1)]

# Create a new histogram column 'count_bin' using cut()
histogram_data_socal <- histogram_data_socal %>%
  mutate(
    count_bin = cut(
      count,
      breaks = bin_edges,
      labels = labels,
      right = FALSE,
      include.lowest = TRUE
    )
  )

# Summarize counts by (count_bin, intersects_fire)
# Drop geometry before grouping
grouped <- histogram_data_socal %>%
  st_drop_geometry() %>%
  group_by(count_bin, intersects_fire) %>%
  summarise(n = n(), .groups = "drop")

# Step 7: Pivot to wide format so we have separate columns for Flood vs. No Flood
pivot <- grouped %>%
  pivot_wider(
    names_from = intersects_fire,
    values_from = n,
    values_fill = list(n = 0)
  ) %>%
  rename(
    `Fire`   = `TRUE`,
    `No fire` = `FALSE`
  ) %>%
  arrange(factor(count_bin, levels = labels))

# (Optional) If you want a long-format data frame for ggplot directly:
pivot_long <- pivot %>%
  pivot_longer(
    cols = c(`No fire`, Fire),
    names_to = "intersects_fire",
    values_to = "n"
  ) %>%
  mutate(
    # Ensure the factor levels match the intended order
    count_bin = factor(count_bin, levels = labels),
    intersects_fire = factor(intersects_fire, levels = c("No fire", "Fire"))
  ) %>% filter(!is.na(count_bin))

# Plot a stacked bar chart with ggplot2
ggplot(pivot_long, aes(x = count_bin, y = n, fill = intersects_fire)) +
  geom_bar(stat = "identity") +
  labs(
    x = "Range of values (disaster-related posts within cell)",
    y = "Number of observed cells",
    title = "Number of disaster-related posts within H3 grid cells\n(2025 Southern California wildfires)",
  ) +
  scale_fill_brewer(
    palette = "Set1",
    name   = "Intersects wildfire area?",
    labels = c(
      "No fire" = "No fire",
      "Fire"    = "Fire"
    )
  ) +
  theme_minimal(base_size = 14) +
  theme(
    text               = element_text(family="barlow", size=14),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.border       = element_blank(),
    plot.title         = element_text(face = "bold", size = 16),
    axis.text          = element_text(size = 13),  # Makes axis labels larger
    axis.title.x       = element_text(size = 14),  # Makes x-axis title larger
    axis.title.y       = element_text(size = 14)   # Makes x-axis title larger
    # panel.grid.minor.y = element_blank(),
  ) -> histogram_plot_socal
histogram_plot_socal

ggsave(
  histogram_plot_socal,
  filename = file.path(FIGURE_PATH, "ggplot", "h3_grid_histogram_socal.pdf"),
  width    = 11, height = 5, units = "in", dpi = 300
)
