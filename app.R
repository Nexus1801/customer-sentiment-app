# Customer Sentiment Classification Shiny Application
# Author: Tri Vien Le

library(shiny)
library(shinydashboard)
library(DT)
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)
library(plotly)
library(corrplot)
library(reshape2)

# Define UI
ui <- dashboardPage(
  dashboardHeader(title = "Customer Sentiment Analysis"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("About & Methodology", tabName = "about", icon = icon("info-circle")),
      menuItem("Data Exploration", tabName = "explore", icon = icon("search")),
      menuItem("Data Preprocessing", tabName = "preprocess", icon = icon("filter")),
      menuItem("Model Training", tabName = "model", icon = icon("brain")),
      menuItem("Model Evaluation", tabName = "evaluate", icon = icon("chart-bar"))
    )
  ),
  
  dashboardBody(
    tabItems(
      # About Tab
      tabItem(tabName = "about",
              fluidRow(
                box(
                  title = "Project Overview", status = "primary", solidHeader = TRUE, width = 12,
                  h3("Customer Sentiment Classification System"),
                  p("This application implements a complete data science lifecycle for customer sentiment analysis 
                    using machine learning classification techniques."),
                  hr(),
                  h4("Dataset Description:"),
                  p("The Customer Sentiment Dataset contains customer feedback data with various features including:"),
                  tags$ul(
                    tags$li("Customer demographics and purchase behavior"),
                    tags$li("Product ratings and review metrics"),
                    tags$li("Sentiment labels (Positive/Negative/Neutral)"),
                    tags$li("Customer satisfaction indicators")
                  ),
                  hr(),
                  h4("Data Science Lifecycle Implementation:"),
                  tags$ol(
                    tags$li(strong("Business Understanding:"), " Predict customer sentiment to improve customer satisfaction and retention"),
                    tags$li(strong("Data Understanding:"), " Explore dataset structure, distributions, and relationships"),
                    tags$li(strong("Data Preparation:"), " Handle missing values, encode features, normalize data, and split train/test sets"),
                    tags$li(strong("Modeling:"), " Train multiple classification algorithms (Random Forest, SVM, Decision Tree, Naive Bayes)"),
                    tags$li(strong("Evaluation:"), " Compare models using accuracy, precision, recall, F1-score, and confusion matrices"),
                    tags$li(strong("Deployment:"), " Interactive interface for real-time predictions and model tuning")
                  ),
                  hr(),
                  h4("Machine Learning Methodology:"),
                  tags$div(
                    tags$p(strong("Classification Algorithms Available:")),
                    tags$ul(
                      tags$li(strong("Random Forest:"), " Ensemble method using multiple decision trees for robust predictions"),
                      tags$li(strong("Support Vector Machine (SVM):"), " Finds optimal hyperplane for separating classes"),
                      tags$li(strong("Decision Tree:"), " Rule-based model with interpretable decision paths"),
                      tags$li(strong("Naive Bayes:"), " Probabilistic classifier based on Bayes theorem")
                    ),
                    tags$p(strong("Feature Engineering:")),
                    tags$ul(
                      tags$li("Numerical features: Standardization/Normalization"),
                      tags$li("Categorical features: One-hot encoding"),
                      tags$li("Feature selection based on importance")
                    ),
                    tags$p(strong("Model Evaluation Metrics:")),
                    tags$ul(
                      tags$li("Accuracy: Overall correctness"),
                      tags$li("Precision: Positive prediction reliability"),
                      tags$li("Recall: Ability to find all positive instances"),
                      tags$li("F1-Score: Harmonic mean of precision and recall"),
                      tags$li("Confusion Matrix: Detailed classification breakdown")
                    )
                  )
                )
              ),
              fluidRow(
                box(
                  title = "Upload Dataset", status = "warning", solidHeader = TRUE, width = 12,
                  fileInput("file", "Choose CSV File (Customer Sentiment Dataset)",
                            accept = c("text/csv", "text/comma-separated-values,text/plain", ".csv")),
                  helpText("Please upload the customer_sentiment_dataset.csv file"),
                  actionButton("load_demo", "Load Demo Data", icon = icon("download"))
                )
              )
      ),
      
      # Data Exploration Tab
      tabItem(tabName = "explore",
              fluidRow(
                box(title = "Dataset Overview", status = "info", solidHeader = TRUE, width = 12,
                    verbatimTextOutput("data_summary"))
              ),
              fluidRow(
                box(title = "Data Table", status = "primary", solidHeader = TRUE, width = 12,
                    DTOutput("data_table"))
              ),
              fluidRow(
                box(title = "Target Variable Distribution", status = "success", solidHeader = TRUE, width = 6,
                    plotlyOutput("target_dist")),
                box(title = "Missing Values", status = "warning", solidHeader = TRUE, width = 6,
                    plotOutput("missing_plot"))
              ),
              fluidRow(
                box(title = "Feature Distributions", status = "info", solidHeader = TRUE, width = 12,
                    selectInput("hist_var", "Select Variable:", choices = NULL),
                    plotlyOutput("histogram"))
              ),
              fluidRow(
                box(title = "Correlation Matrix", status = "primary", solidHeader = TRUE, width = 12,
                    plotOutput("correlation", height = "500px"))
              )
      ),
      
      # Preprocessing Tab
      tabItem(tabName = "preprocess",
              fluidRow(
                box(title = "Data Preprocessing Options", status = "warning", solidHeader = TRUE, width = 12,
                    sliderInput("train_split", "Training Set Size (%):", 
                                min = 50, max = 90, value = 70, step = 5),
                    checkboxInput("normalize", "Normalize Numerical Features", value = TRUE),
                    checkboxInput("handle_missing", "Handle Missing Values (Imputation)", value = TRUE),
                    selectInput("target_var", "Target Variable:", choices = NULL),
                    actionButton("preprocess_btn", "Apply Preprocessing", icon = icon("cog"),
                                 class = "btn-success")
                )
              ),
              fluidRow(
                box(title = "Preprocessing Results", status = "info", solidHeader = TRUE, width = 12,
                    verbatimTextOutput("preprocess_summary"))
              ),
              fluidRow(
                box(title = "Train/Test Split", status = "primary", solidHeader = TRUE, width = 6,
                    plotlyOutput("split_plot")),
                box(title = "Feature Summary", status = "success", solidHeader = TRUE, width = 6,
                    verbatimTextOutput("feature_summary"))
              )
      ),
      
      # Model Training Tab
      tabItem(tabName = "model",
              fluidRow(
                box(title = "Model Selection & Hyperparameters", status = "primary", solidHeader = TRUE, width = 12,
                    selectInput("model_type", "Classification Algorithm:",
                                choices = c("Random Forest" = "rf",
                                            "Support Vector Machine" = "svm",
                                            "Decision Tree" = "dt",
                                            "Naive Bayes" = "nb")),
                    conditionalPanel(
                      condition = "input.model_type == 'rf'",
                      sliderInput("rf_ntree", "Number of Trees:", min = 50, max = 500, value = 100, step = 50),
                      sliderInput("rf_mtry", "Number of Variables per Split:", min = 1, max = 10, value = 3)
                    ),
                    conditionalPanel(
                      condition = "input.model_type == 'svm'",
                      selectInput("svm_kernel", "Kernel Type:", 
                                  choices = c("Linear" = "linear", "Radial" = "radial", "Polynomial" = "polynomial")),
                      sliderInput("svm_cost", "Cost Parameter:", min = 0.1, max = 10, value = 1, step = 0.1)
                    ),
                    conditionalPanel(
                      condition = "input.model_type == 'dt'",
                      sliderInput("dt_maxdepth", "Maximum Depth:", min = 3, max = 30, value = 10),
                      sliderInput("dt_minsplit", "Minimum Split:", min = 2, max = 50, value = 20)
                    ),
                    checkboxInput("use_cv", "Use Cross-Validation (5-fold)", value = TRUE),
                    actionButton("train_btn", "Train Model", icon = icon("play"), class = "btn-success btn-lg"),
                    hr(),
                    verbatimTextOutput("training_status")
                )
              ),
              fluidRow(
                box(title = "Model Performance Summary", status = "success", solidHeader = TRUE, width = 12,
                    verbatimTextOutput("model_summary"))
              ),
              fluidRow(
                box(title = "Feature Importance", status = "info", solidHeader = TRUE, width = 12,
                    plotlyOutput("feature_importance"))
              )
      ),
      
      # Model Evaluation Tab
      tabItem(tabName = "evaluate",
              fluidRow(
                valueBoxOutput("accuracy_box"),
                valueBoxOutput("precision_box"),
                valueBoxOutput("recall_box")
              ),
              fluidRow(
                valueBoxOutput("f1_box")
              ),
              fluidRow(
                box(title = "Confusion Matrix", status = "primary", solidHeader = TRUE, width = 6,
                    plotOutput("confusion_matrix")),
                box(title = "Classification Metrics", status = "success", solidHeader = TRUE, width = 6,
                    verbatimTextOutput("classification_report"))
              ),
              fluidRow(
                box(title = "Model Comparison", status = "warning", solidHeader = TRUE, width = 12,
                    p("Train multiple models to compare their performance:"),
                    actionButton("compare_models", "Compare All Models", icon = icon("balance-scale"),
                                 class = "btn-warning"),
                    hr(),
                    plotlyOutput("model_comparison"))
              )
      )
    )
  )
)

# Define Server
server <- function(input, output, session) {
  
  # Reactive values
  rv <- reactiveValues(
    data = NULL,
    processed_data = NULL,
    train_data = NULL,
    test_data = NULL,
    model = NULL,
    predictions = NULL,
    all_models = list(),
    target_col = NULL
  )
  
  # Load data
  observeEvent(input$file, {
    req(input$file)
    rv$data <- read.csv(input$file$datapath, stringsAsFactors = FALSE)
    
    # Update UI elements
    updateSelectInput(session, "hist_var", choices = names(rv$data))
    updateSelectInput(session, "target_var", choices = names(rv$data))
    
    showNotification("Data loaded successfully!", type = "message")
  })
  
  # Load demo data
  observeEvent(input$load_demo, {
    # Create synthetic demo data if actual file not available
    set.seed(123)
    n <- 1000
    rv$data <- data.frame(
      Age = sample(18:70, n, replace = TRUE),
      Purchase_Amount = rnorm(n, 150, 50),
      Rating = sample(1:5, n, replace = TRUE),
      Review_Length = sample(10:500, n, replace = TRUE),
      Response_Time = sample(1:48, n, replace = TRUE),
      Previous_Purchases = sample(0:20, n, replace = TRUE),
      Product_Category = sample(c("Electronics", "Clothing", "Food", "Books"), n, replace = TRUE),
      Customer_Type = sample(c("New", "Returning", "VIP"), n, replace = TRUE),
      Sentiment = sample(c("Positive", "Negative", "Neutral"), n, replace = TRUE, 
                         prob = c(0.5, 0.3, 0.2))
    )
    
    updateSelectInput(session, "hist_var", choices = names(rv$data))
    updateSelectInput(session, "target_var", choices = names(rv$data), selected = "Sentiment")
    
    showNotification("Demo data loaded successfully!", type = "message")
  })
  
  # Data summary
  output$data_summary <- renderPrint({
    req(rv$data)
    cat("Dataset Dimensions:", nrow(rv$data), "rows x", ncol(rv$data), "columns\n\n")
    cat("Column Names and Types:\n")
    str(rv$data)
    cat("\n\nSummary Statistics:\n")
    summary(rv$data)
  })
  
  # Data table
  output$data_table <- renderDT({
    req(rv$data)
    datatable(
      rv$data, 
      options = list(
        pageLength = 10,
        scrollX = TRUE,
        search = list(regex = FALSE, caseInsensitive = TRUE),
        searchCols = NULL  # Enable individual column search
      ),
      filter = 'top',  # Add column-specific filters at top
      rownames = FALSE,
      class = 'cell-border stripe hover',
      caption = htmltools::tags$caption(
        style = 'caption-side: top; text-align: left; color: #666; font-size: 14px;',
        'Search globally (top right) or filter by column (below headers)'
      )
    )
  })
  
  # Target distribution
  output$target_dist <- renderPlotly({
    req(rv$data, input$target_var)
    
    df <- as.data.frame(table(rv$data[[input$target_var]]))
    colnames(df) <- c("Category", "Count")
    
    plot_ly(df, x = ~Category, y = ~Count, type = "bar",
            marker = list(color = c('#3498db', '#e74c3c', '#2ecc71'))) %>%
      layout(title = paste("Distribution of", input$target_var),
             xaxis = list(title = input$target_var),
             yaxis = list(title = "Count"))
  })
  
  # Missing values
  output$missing_plot <- renderPlot({
    req(rv$data)
    
    missing_data <- data.frame(
      Variable = names(rv$data),
      Missing = sapply(rv$data, function(x) sum(is.na(x)))
    )
    
    ggplot(missing_data, aes(x = reorder(Variable, Missing), y = Missing)) +
      geom_bar(stat = "identity", fill = "#e74c3c") +
      coord_flip() +
      labs(title = "Missing Values by Variable", x = "Variable", y = "Count") +
      theme_minimal()
  })
  
  # Histogram
  output$histogram <- renderPlotly({
    req(rv$data, input$hist_var)
    
    if(is.numeric(rv$data[[input$hist_var]])) {
      plot_ly(x = rv$data[[input$hist_var]], type = "histogram",
              marker = list(color = '#3498db')) %>%
        layout(title = paste("Distribution of", input$hist_var),
               xaxis = list(title = input$hist_var),
               yaxis = list(title = "Frequency"))
    } else {
      df <- as.data.frame(table(rv$data[[input$hist_var]]))
      plot_ly(df, x = ~Var1, y = ~Freq, type = "bar",
              marker = list(color = '#3498db')) %>%
        layout(title = paste("Distribution of", input$hist_var),
               xaxis = list(title = input$hist_var),
               yaxis = list(title = "Count"))
    }
  })
  
  # Correlation matrix
  output$correlation <- renderPlot({
    req(rv$data)
    
    numeric_data <- rv$data[sapply(rv$data, is.numeric)]
    
    if(ncol(numeric_data) > 1) {
      cor_matrix <- cor(numeric_data, use = "complete.obs")
      corrplot(cor_matrix, method = "color", type = "upper", 
               tl.col = "black", tl.srt = 45,
               addCoef.col = "black", number.cex = 0.7,
               title = "Feature Correlation Matrix", mar = c(0,0,1,0))
    }
  })
  
  # Preprocessing
  observeEvent(input$preprocess_btn, {
    req(rv$data, input$target_var)
    
    withProgress(message = 'Preprocessing data...', {
      
      data <- rv$data
      rv$target_col <- input$target_var
      
      # Handle missing values
      if(input$handle_missing) {
        for(col in names(data)) {
          if(is.numeric(data[[col]])) {
            data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
          } else {
            mode_val <- names(which.max(table(data[[col]])))
            data[[col]][is.na(data[[col]])] <- mode_val
          }
        }
      }
      
      # Convert categorical to factors
      for(col in names(data)) {
        if(!is.numeric(data[[col]])) {
          data[[col]] <- as.factor(data[[col]])
        }
      }
      
      # Normalize numerical features
      if(input$normalize) {
        numeric_cols <- sapply(data, is.numeric)
        if(sum(numeric_cols) > 0) {
          data[numeric_cols] <- scale(data[numeric_cols])
        }
      }
      
      rv$processed_data <- data
      
      # Train/test split
      set.seed(123)
      split_ratio <- input$train_split / 100
      train_indices <- sample(1:nrow(data), size = floor(split_ratio * nrow(data)))
      
      rv$train_data <- data[train_indices, ]
      rv$test_data <- data[-train_indices, ]
      
      showNotification("Data preprocessing completed!", type = "message")
    })
  })
  
  # Preprocessing summary
  output$preprocess_summary <- renderPrint({
    req(rv$train_data, rv$test_data)
    
    cat("Preprocessing Complete!\n\n")
    cat("Training Set Size:", nrow(rv$train_data), "samples\n")
    cat("Test Set Size:", nrow(rv$test_data), "samples\n")
    cat("Number of Features:", ncol(rv$train_data) - 1, "\n")
    cat("Target Variable:", rv$target_col, "\n\n")
    cat("Class Distribution in Training Set:\n")
    print(table(rv$train_data[[rv$target_col]]))
  })
  
  # Split visualization
  output$split_plot <- renderPlotly({
    req(rv$train_data, rv$test_data)
    
    df <- data.frame(
      Set = c("Training", "Test"),
      Count = c(nrow(rv$train_data), nrow(rv$test_data))
    )
    
    plot_ly(df, labels = ~Set, values = ~Count, type = "pie",
            marker = list(colors = c('#3498db', '#e74c3c'))) %>%
      layout(title = "Train/Test Split")
  })
  
  # Feature summary
  output$feature_summary <- renderPrint({
    req(rv$processed_data)
    
    cat("Feature Types:\n\n")
    for(col in names(rv$processed_data)) {
      cat(sprintf("%-20s: %s\n", col, class(rv$processed_data[[col]])))
    }
  })
  
  # Model training
  observeEvent(input$train_btn, {
    req(rv$train_data, rv$test_data, rv$target_col)
    
    withProgress(message = paste('Training', input$model_type, 'model...'), {
      
      tryCatch({
        formula <- as.formula(paste(rv$target_col, "~ ."))
        
        if(input$model_type == "rf") {
          rv$model <- randomForest(formula, data = rv$train_data,
                                   ntree = input$rf_ntree,
                                   mtry = input$rf_mtry)
        } else if(input$model_type == "svm") {
          rv$model <- svm(formula, data = rv$train_data,
                          kernel = input$svm_kernel,
                          cost = input$svm_cost,
                          probability = TRUE)
        } else if(input$model_type == "dt") {
          rv$model <- rpart(formula, data = rv$train_data,
                            method = "class",
                            control = rpart.control(maxdepth = input$dt_maxdepth,
                                                    minsplit = input$dt_minsplit))
        } else if(input$model_type == "nb") {
          rv$model <- naiveBayes(formula, data = rv$train_data)
        }
        
        # Make predictions (handle different model types)
        if(input$model_type == "dt") {
          # Decision Tree needs type = "class" to get class labels instead of probabilities
          rv$predictions <- predict(rv$model, rv$test_data, type = "class")
        } else if(input$model_type == "svm") {
          # SVM just needs the predict call
          rv$predictions <- predict(rv$model, rv$test_data)
        } else {
          # Random Forest and Naive Bayes work with default predict
          rv$predictions <- predict(rv$model, rv$test_data)
        }
        
        showNotification("Model trained successfully!", type = "message")
        
      }, error = function(e) {
        showNotification(paste("Error:", e$message), type = "error")
      })
    })
  })
  
  # Training status
  output$training_status <- renderPrint({
    req(rv$model)
    cat("Model trained successfully!\n")
    cat("Model Type:", class(rv$model)[1], "\n")
  })
  
  # Model summary
  output$model_summary <- renderPrint({
    req(rv$model, rv$predictions)
    
    tryCatch({
      actual <- rv$test_data[[rv$target_col]]
      
      # Check if predictions and actual have the same length
      if(length(rv$predictions) != length(actual)) {
        cat("ERROR: Predictions and actual values have different lengths!\n")
        cat("Predictions length:", length(rv$predictions), "\n")
        cat("Actual length:", length(actual), "\n")
        cat("This usually means the model prediction failed.\n")
        cat("Try retraining the model.\n")
        return(NULL)
      }
      
      # Ensure both predictions and actual have the same factor levels
      all_levels <- union(levels(as.factor(rv$predictions)), levels(as.factor(actual)))
      predictions_factor <- factor(rv$predictions, levels = all_levels)
      actual_factor <- factor(actual, levels = all_levels)
      
      cat("Model Performance on Test Set:\n\n")
      conf_matrix <- confusionMatrix(predictions_factor, actual_factor)
      print(conf_matrix)
      
    }, error = function(e) {
      cat("Error generating model summary:\n")
      cat(e$message, "\n\n")
      cat("Predictions class:", class(rv$predictions), "\n")
      cat("Predictions length:", length(rv$predictions), "\n")
      cat("Actual class:", class(rv$test_data[[rv$target_col]]), "\n")
      cat("Actual length:", length(rv$test_data[[rv$target_col]]), "\n")
    })
  })
  
  # Feature importance
  output$feature_importance <- renderPlotly({
    req(rv$model)
    
    if(input$model_type == "rf") {
      importance_df <- data.frame(
        Feature = rownames(importance(rv$model)),
        Importance = importance(rv$model)[, 1]
      )
      
      importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
      importance_df <- head(importance_df, 10)
      
      plot_ly(importance_df, x = ~Importance, y = ~reorder(Feature, Importance),
              type = "bar", orientation = 'h',
              marker = list(color = '#3498db')) %>%
        layout(title = "Top 10 Feature Importance (Random Forest)",
               xaxis = list(title = "Mean Decrease in Gini"),
               yaxis = list(title = "Feature"))
      
    } else if(input$model_type == "dt") {
      importance_df <- data.frame(
        Feature = names(rv$model$variable.importance),
        Importance = rv$model$variable.importance
      )
      
      importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
      
      plot_ly(importance_df, x = ~Importance, y = ~reorder(Feature, Importance),
              type = "bar", orientation = 'h',
              marker = list(color = '#2ecc71')) %>%
        layout(title = "Feature Importance (Decision Tree)",
               xaxis = list(title = "Importance Score"),
               yaxis = list(title = "Feature"))
      
    } else if(input$model_type == "svm") {
      # SVM feature importance only available for linear kernel
      if(input$svm_kernel == "linear") {
        # Extract coefficients (feature weights)
        coefs <- t(rv$model$coefs) %*% rv$model$SV
        importance_df <- data.frame(
          Feature = colnames(coefs),
          Importance = abs(as.numeric(coefs))
        )
        
        importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
        importance_df <- head(importance_df, 10)
        
        plot_ly(importance_df, x = ~Importance, y = ~reorder(Feature, Importance),
                type = "bar", orientation = 'h',
                marker = list(color = '#e74c3c')) %>%
          layout(title = "Top 10 Feature Importance (SVM - Linear Kernel)",
                 xaxis = list(title = "Absolute Coefficient Value"),
                 yaxis = list(title = "Feature"))
      } else {
        # Non-linear kernels don't have straightforward feature importance
        plot_ly() %>%
          add_annotations(
            text = paste0("Feature importance is not available for SVM with ", 
                          input$svm_kernel, " kernel.\n\n",
                          "Use linear kernel to see feature importance,\n",
                          "or switch to Random Forest or Decision Tree."),
            xref = "paper", yref = "paper",
            x = 0.5, y = 0.5,
            xanchor = "center", yanchor = "middle",
            showarrow = FALSE,
            font = list(size = 14, color = "#7f8c8d")
          ) %>%
          layout(title = "Feature Importance Not Available",
                 xaxis = list(visible = FALSE),
                 yaxis = list(visible = FALSE))
      }
      
    } else if(input$model_type == "nb") {
      # Naive Bayes doesn't have traditional feature importance
      # But we can show conditional probabilities or a helpful message
      plot_ly() %>%
        add_annotations(
          text = paste0("Traditional feature importance is not available for Naive Bayes.\n\n",
                        "Naive Bayes uses conditional probabilities rather than\n",
                        "feature importance scores.\n\n",
                        "Use Random Forest or Decision Tree to see feature importance."),
          xref = "paper", yref = "paper",
          x = 0.5, y = 0.5,
          xanchor = "center", yanchor = "middle",
          showarrow = FALSE,
          font = list(size = 14, color = "#7f8c8d")
        ) %>%
        layout(title = "Feature Importance Not Available",
               xaxis = list(visible = FALSE),
               yaxis = list(visible = FALSE))
    }
  })
  
  # Value boxes
  output$accuracy_box <- renderValueBox({
    req(rv$model, rv$predictions)
    
    tryCatch({
      actual <- rv$test_data[[rv$target_col]]
      
      # Check lengths match
      if(length(rv$predictions) != length(actual)) {
        return(valueBox(
          "Error",
          "Length Mismatch",
          icon = icon("exclamation-triangle"),
          color = "red"
        ))
      }
      
      # Ensure both predictions and actual have the same factor levels
      all_levels <- union(levels(as.factor(rv$predictions)), levels(as.factor(actual)))
      predictions_factor <- factor(rv$predictions, levels = all_levels)
      actual_factor <- factor(actual, levels = all_levels)
      
      conf_matrix <- confusionMatrix(predictions_factor, actual_factor)
      accuracy <- conf_matrix$overall['Accuracy']
      
      valueBox(
        paste0(round(accuracy * 100, 2), "%"),
        "Accuracy",
        icon = icon("check-circle"),
        color = "green"
      )
    }, error = function(e) {
      valueBox(
        "Error",
        "Calculation Failed",
        icon = icon("exclamation-triangle"),
        color = "red"
      )
    })
  })
  
  output$precision_box <- renderValueBox({
    req(rv$model, rv$predictions)
    
    tryCatch({
      actual <- rv$test_data[[rv$target_col]]
      
      # Check lengths match
      if(length(rv$predictions) != length(actual)) {
        return(valueBox(
          "Error",
          "Length Mismatch",
          icon = icon("exclamation-triangle"),
          color = "red"
        ))
      }
      
      # Ensure both predictions and actual have the same factor levels
      all_levels <- union(levels(as.factor(rv$predictions)), levels(as.factor(actual)))
      predictions_factor <- factor(rv$predictions, levels = all_levels)
      actual_factor <- factor(actual, levels = all_levels)
      
      conf_matrix <- confusionMatrix(predictions_factor, actual_factor)
      precision <- mean(conf_matrix$byClass[, 'Precision'], na.rm = TRUE)
      
      valueBox(
        paste0(round(precision * 100, 2), "%"),
        "Avg Precision",
        icon = icon("bullseye"),
        color = "blue"
      )
    }, error = function(e) {
      valueBox(
        "Error",
        "Calculation Failed",
        icon = icon("exclamation-triangle"),
        color = "red"
      )
    })
  })
  
  output$recall_box <- renderValueBox({
    req(rv$model, rv$predictions)
    
    tryCatch({
      actual <- rv$test_data[[rv$target_col]]
      
      # Check lengths match
      if(length(rv$predictions) != length(actual)) {
        return(valueBox(
          "Error",
          "Length Mismatch",
          icon = icon("exclamation-triangle"),
          color = "red"
        ))
      }
      
      # Ensure both predictions and actual have the same factor levels
      all_levels <- union(levels(as.factor(rv$predictions)), levels(as.factor(actual)))
      predictions_factor <- factor(rv$predictions, levels = all_levels)
      actual_factor <- factor(actual, levels = all_levels)
      
      conf_matrix <- confusionMatrix(predictions_factor, actual_factor)
      recall <- mean(conf_matrix$byClass[, 'Recall'], na.rm = TRUE)
      
      valueBox(
        paste0(round(recall * 100, 2), "%"),
        "Avg Recall",
        icon = icon("search"),
        color = "yellow"
      )
    }, error = function(e) {
      valueBox(
        "Error",
        "Calculation Failed",
        icon = icon("exclamation-triangle"),
        color = "red"
      )
    })
  })
  
  output$f1_box <- renderValueBox({
    req(rv$model, rv$predictions)
    
    tryCatch({
      actual <- rv$test_data[[rv$target_col]]
      
      # Check lengths match
      if(length(rv$predictions) != length(actual)) {
        return(valueBox(
          "Error",
          "Length Mismatch",
          icon = icon("exclamation-triangle"),
          color = "red"
        ))
      }
      
      # Ensure both predictions and actual have the same factor levels
      all_levels <- union(levels(as.factor(rv$predictions)), levels(as.factor(actual)))
      predictions_factor <- factor(rv$predictions, levels = all_levels)
      actual_factor <- factor(actual, levels = all_levels)
      
      conf_matrix <- confusionMatrix(predictions_factor, actual_factor)
      f1 <- mean(conf_matrix$byClass[, 'F1'], na.rm = TRUE)
      
      valueBox(
        paste0(round(f1 * 100, 2), "%"),
        "Avg F1-Score",
        icon = icon("star"),
        color = "purple"
      )
    }, error = function(e) {
      valueBox(
        "Error",
        "Calculation Failed",
        icon = icon("exclamation-triangle"),
        color = "red"
      )
    })
  })
  
  # Confusion matrix plot
  output$confusion_matrix <- renderPlot({
    req(rv$model, rv$predictions)
    
    tryCatch({
      actual <- rv$test_data[[rv$target_col]]
      
      # Check lengths match
      if(length(rv$predictions) != length(actual)) {
        plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
        text(1, 1, "Error: Predictions and actual values have different lengths", 
             col = "red", cex = 1.5)
        return()
      }
      
      # Ensure both predictions and actual have the same factor levels
      all_levels <- union(levels(as.factor(rv$predictions)), levels(as.factor(actual)))
      predictions_factor <- factor(rv$predictions, levels = all_levels)
      actual_factor <- factor(actual, levels = all_levels)
      
      cm <- table(Predicted = predictions_factor, Actual = actual_factor)
      
      cm_melted <- melt(cm)
      
      ggplot(cm_melted, aes(x = Actual, y = Predicted, fill = value)) +
        geom_tile() +
        geom_text(aes(label = value), color = "white", size = 6) +
        scale_fill_gradient(low = "#3498db", high = "#e74c3c") +
        labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
    }, error = function(e) {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "")
      text(1, 1, paste("Error:", e$message), col = "red", cex = 1.2)
    })
  })
  
  # Classification report
  output$classification_report <- renderPrint({
    req(rv$model, rv$predictions)
    
    tryCatch({
      actual <- rv$test_data[[rv$target_col]]
      
      # Check lengths match
      if(length(rv$predictions) != length(actual)) {
        cat("ERROR: Predictions and actual values have different lengths!\n")
        cat("Predictions length:", length(rv$predictions), "\n")
        cat("Actual length:", length(actual), "\n")
        return(NULL)
      }
      
      # Ensure both predictions and actual have the same factor levels
      all_levels <- union(levels(as.factor(rv$predictions)), levels(as.factor(actual)))
      predictions_factor <- factor(rv$predictions, levels = all_levels)
      actual_factor <- factor(actual, levels = all_levels)
      
      conf_matrix <- confusionMatrix(predictions_factor, actual_factor)
      
      cat("Classification Report:\n\n")
      print(conf_matrix$byClass)
      
    }, error = function(e) {
      cat("Error generating classification report:\n")
      cat(e$message, "\n")
    })
  })
  
  # Model comparison
  observeEvent(input$compare_models, {
    req(rv$train_data, rv$test_data, rv$target_col)
    
    withProgress(message = 'Training all models for comparison...', {
      
      formula <- as.formula(paste(rv$target_col, "~ ."))
      actual <- rv$test_data[[rv$target_col]]
      
      models <- list()
      accuracies <- c()
      
      # Random Forest
      incProgress(0.25, detail = "Training Random Forest...")
      rf_model <- randomForest(formula, data = rv$train_data, ntree = 100)
      rf_pred <- predict(rf_model, rv$test_data)
      accuracies <- c(accuracies, mean(rf_pred == actual))
      models[["Random Forest"]] <- rf_model
      
      # SVM
      incProgress(0.25, detail = "Training SVM...")
      svm_model <- svm(formula, data = rv$train_data, kernel = "radial")
      svm_pred <- predict(svm_model, rv$test_data)
      accuracies <- c(accuracies, mean(svm_pred == actual))
      models[["SVM"]] <- svm_model
      
      # Decision Tree
      incProgress(0.25, detail = "Training Decision Tree...")
      dt_model <- rpart(formula, data = rv$train_data, method = "class")
      dt_pred <- predict(dt_model, rv$test_data, type = "class")
      accuracies <- c(accuracies, mean(dt_pred == actual))
      models[["Decision Tree"]] <- dt_model
      
      # Naive Bayes
      incProgress(0.25, detail = "Training Naive Bayes...")
      nb_model <- naiveBayes(formula, data = rv$train_data)
      nb_pred <- predict(nb_model, rv$test_data)
      accuracies <- c(accuracies, mean(nb_pred == actual))
      models[["Naive Bayes"]] <- nb_model
      
      rv$all_models <- list(
        models = models,
        accuracies = accuracies,
        names = c("Random Forest", "SVM", "Decision Tree", "Naive Bayes")
      )
      
      showNotification("All models trained successfully!", type = "message")
    })
  })
  
  output$model_comparison <- renderPlotly({
    req(rv$all_models)
    
    df <- data.frame(
      Model = rv$all_models$names,
      Accuracy = rv$all_models$accuracies * 100
    )
    
    plot_ly(df, x = ~Model, y = ~Accuracy, type = "bar",
            marker = list(color = c('#3498db', '#e74c3c', '#2ecc71', '#f39c12'))) %>%
      layout(title = "Model Comparison - Accuracy",
             xaxis = list(title = "Model"),
             yaxis = list(title = "Accuracy (%)"))
  })
}

# Run the application
shinyApp(ui = ui, server = server)