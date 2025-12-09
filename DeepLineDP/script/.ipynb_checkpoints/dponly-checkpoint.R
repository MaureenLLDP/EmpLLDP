library(tidyverse)
library(pROC)

get.top.k.tokens = function(df, k)
{
  top.k <- df %>% filter( is.comment.line=="False"  & file.level.ground.truth=="True" & prediction.label=="True" ) %>%
    group_by(test, filename) %>% top_n(k, token.attention.score) %>% select("project","train","test","filename","token") %>% distinct()
  
  top.k$flag = 'topk'

  return(top.k)
}

# ============ Within-Release Performance ============

prediction_dir = '../output/prediction/DeepLineDP/within-release/'

all_files = list.files(prediction_dir)

df_all <- NULL

for(f in all_files)
{
  df <- read.csv(paste0(prediction_dir, f))
  df_all <- rbind(df_all, df)
}

#Force attention score of comment line is 0
df_all[df_all$is.comment.line == "True",]$token.attention.score = 0

tmp.top.k = get.top.k.tokens(df_all, 1500)

merged_df_all = merge(df_all, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)

merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0

## use top-k tokens 
sum_line_attn = merged_df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(test, filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
  summarize(attention_score = sum(token.attention.score), num_tokens = n())

sorted = sum_line_attn %>% group_by(test, filename) %>% arrange(-attention_score, .by_group=TRUE) %>% mutate(order = row_number())

# calculate IFA
IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(test, filename) %>% top_n(1, -order)

total_true = sorted %>% group_by(test, filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))

# calculate Recall20%LOC
recall20LOC = sorted %>% group_by(test, filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
  summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
  merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)

# calculate Effort20%Recall
effort20Recall = sorted %>% merge(total_true) %>% group_by(test, filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
  summarise(effort20Recall = sum(recall <= 0.2)/n())

# calculate AUC for line-level
line.auc = sorted %>% group_by(test, filename) %>%
  summarise(AUC = as.numeric(pROC::auc(line.level.ground.truth, attention_score)))

## get within-project result
IFA$project = str_replace(IFA$test, '-.*','')
recall20LOC$project = str_replace(recall20LOC$test, '-.*','')
effort20Recall$project = str_replace(effort20Recall$test, '-.*','')
line.auc$project = str_replace(line.auc$test, '-.*','')

ifa.each.project = IFA %>% group_by(project) %>% summarise(mean.by.project = mean(order))
recall.each.project = recall20LOC %>% group_by(project) %>% summarise(mean.by.project = mean(recall20LOC))
effort.each.project = effort20Recall %>% group_by(project) %>% summarise(mean.by.project = mean(effort20Recall))
auc.each.project = line.auc %>% group_by(project) %>% summarise(mean.by.project = mean(AUC))

within.line.level.by.project = data.frame(
  project = ifa.each.project$project, 
  IFA = ifa.each.project$mean.by.project, 
  `Recall@20%LOC` = recall.each.project$mean.by.project, 
  `Effort@20%Recall` = effort.each.project$mean.by.project, 
  AUC = auc.each.project$mean.by.project,
  check.names = FALSE
)

# Save within-release results
write.csv(within.line.level.by.project, "../output/within_release_line_level_performance.csv", row.names = FALSE)

cat("Within-release line-level performance saved to: ../output/within_release_line_level_performance.csv\n")
print(within.line.level.by.project)


# ============ Cross-Release Performance ============

get.line.level.metrics = function(df_all)
{
  #Force attention score of comment line is 0
  df_all[df_all$is.comment.line == "True",]$token.attention.score = 0

  sum_line_attn = df_all %>% filter(file.level.ground.truth == "True" & prediction.label == "True") %>% group_by(filename,is.comment.line, file.level.ground.truth, prediction.label, line.number, line.level.ground.truth) %>%
    summarize(attention_score = sum(token.attention.score), num_tokens = n())
  sorted = sum_line_attn %>% group_by(filename) %>% arrange(-attention_score, .by_group=TRUE) %>% mutate(order = row_number())
  
  # calculate IFA
  IFA = sorted %>% filter(line.level.ground.truth == "True") %>% group_by(filename) %>% top_n(1, -order)
  total_true = sorted %>% group_by(filename) %>% summarize(total_true = sum(line.level.ground.truth == "True"))
  
  # calculate Recall20%LOC
  recall20LOC = sorted %>% group_by(filename) %>% mutate(effort = round(order/n(),digits = 2 )) %>% filter(effort <= 0.2) %>%
    summarize(correct_pred = sum(line.level.ground.truth == "True")) %>%
    merge(total_true) %>% mutate(recall20LOC = correct_pred/total_true)

  # calculate Effort20%Recall
  effort20Recall = sorted %>% merge(total_true) %>% group_by(filename) %>% mutate(cummulative_correct_pred = cumsum(line.level.ground.truth == "True"), recall = round(cumsum(line.level.ground.truth == "True")/total_true, digits = 2)) %>%
    summarise(effort20Recall = sum(recall <= 0.2)/n())
  
  # calculate AUC
  line.auc = sorted %>% group_by(filename) %>%
    summarise(AUC = as.numeric(pROC::auc(line.level.ground.truth, attention_score)))
  
  all.ifa = IFA$order
  all.recall = recall20LOC$recall20LOC
  all.effort = effort20Recall$effort20Recall
  all.auc = line.auc$AUC
  
  result.df = data.frame(all.ifa, all.recall, all.effort, all.auc)
  
  return(result.df)
}

prediction.dir = '../output/prediction/DeepLineDP/cross-project/'

projs = c('activemq', 'camel', 'derby', 'groovy', 'hbase', 'hive', 'jruby', 'lucene', 'wicket')

all.line.result = NULL

for(p in projs)
{
  actual.pred.dir = paste0(prediction.dir,p,'/')
  
  if(!dir.exists(actual.pred.dir)) {
    cat(paste0('Warning: Directory does not exist: ', actual.pred.dir, '\n'))
    next
  }
  
  all.files = list.files(actual.pred.dir)
  
  if(length(all.files) == 0) {
    cat(paste0('Warning: No files found in: ', actual.pred.dir, '\n'))
    next
  }
  
  for(f in all.files)
  {
    df = read.csv(paste0(actual.pred.dir,f))

    f = str_replace(f,'.csv','')
    f.split = unlist(strsplit(f,'-'))
    target = tail(f.split,2)[1]

    tmp.top.k = get.top.k.tokens(df, 1500)
    
    merged_df_all = merge(df, tmp.top.k, by=c('project', 'train', 'test', 'filename', 'token'), all.x = TRUE)
    
    merged_df_all[is.na(merged_df_all$flag),]$token.attention.score = 0
    
    line.level.result = get.line.level.metrics(merged_df_all)
    line.level.result$src = p
    line.level.result$target = target

    all.line.result = rbind(all.line.result, line.level.result)

    print(paste0('finished ',f))
  }

  print(paste0('finished ',p))
}

if(is.null(all.line.result)) {
  cat("\n\nError: No cross-release data was processed. Please check if the cross-release directory exists and contains data.\n")
} else {
  cross.line.level.by.project = all.line.result %>% group_by(target) %>% 
    summarize(
      IFA = mean(all.ifa), 
      `Recall@20%LOC` = mean(all.recall), 
      `Effort@20%Recall` = mean(all.effort), 
      AUC = mean(all.auc)
    )

  names(cross.line.level.by.project)[1] = "project"

  # Save cross-release results
  write.csv(cross.line.level.by.project, "../output/cross_release_line_level_performance.csv", row.names = FALSE)

  cat("\n\nCross-release line-level performance saved to: ../output/cross_release_line_level_performance.csv\n")
  print(cross.line.level.by.project)
}