# library(shiny)
packs = c("xgboost", "caret", "RCurl", "jsonlite", "dplyr", "naniar")

if (length(setdiff(packs, rownames(installed.packages())))>0) {
  suppressMessages(install.packages(setdiff(packs, 
                                            rownames(installed.packages())),
                                    repos='http://cran.uni-muenster.de/'))
}
suppressMessages(invisible(lapply(packs, require, character.only=TRUE)))
suppressMessages(rm(packs))


################################################################################
token = NULL
uri =  NULL
################################################################################


#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
model <- readRDS("model.rds")

id <- postForm(
  uri= uri,
  token= token,
  content ='record',
  format ='json',
  type ='flat',
  'fields[2]'='record_id',
  rawOrLabel ='raw',
  rawOrLabelHeaders ='raw',
  exportCheckboxLabel ='false',
  exportSurveyFields ='false',
  exportDataAccessGroups='false',
  returnFormat='json'
)
id <- fromJSON(id)

# Define UI ----
ui <- fluidPage (
  titlePanel("PerPain"),
  sidebarLayout(
    sidebarPanel(
      selectInput("select", label = h3("Record ID"), 
                  choices = id$record_id, 
                  selected = 14),
      hr(),
      plotOutput(outputId = "distPlot"),
      textOutput("result")
    ),
    mainPanel(
      textOutput("Pred")
    )       
  )
)

names = c("DYSFUNCTIONAL", "DISTRESSED", "COPERS")
variables = c('PS', 'I', 'LC', 'AD', 'S', 'PR', 'SR', 'DR', 'GA')
missing_val = -99
# Define server logic ----
server <- function(input, output) {
  
  data1 <- reactive({fromJSON(postForm(
    uri = uri,
    token = token ,
    content = 'record',
    format ='json',
    type = 'flat',
    'records[0]'= req(input$select),
    'fields[0]'='geschlecht',
    'fields[1]'='alter',
    'forms[0]'='mpid',
    'events[0]'= 't0_questionnaires_arm_1',
    rawOrLabel ='raw',
    rawOrLabelHeaders ='raw',
    exportCheckboxLabel='false',
    exportSurveyFields ='false',
    exportDataAccessGroups ='false',
    returnFormat ='json'
  ))})
  
# create a data frame for input into a model
  mpi_f <- reactive(req(as.numeric(format(mean(as.numeric(c(
    data1()$mpid_3_3,
    data1()$mpid_3_6,
    data1()$mpid_3_7,
    data1()$mpid_3_12,
    data1()$mpid_3_13,
    data1()$mpid_3_16,
    data1()$mpid_3_17,
    data1()$mpid_3_18))),
    digits = 3))))
  
  mpi_h = reactive(req(as.numeric(format(mean(as.numeric(c(
    data1()$mpid_3_1,
    data1()$mpid_3_4,
    data1()$mpid_3_8,
    data1()$mpid_3_10,
    data1()$mpid_3_14))),
    digits = 3))))
  
  # The case with possibly missing values
  mpi_o = reactive(req(as.numeric(format(mean(c(
    ifelse(as.numeric(data1()$mpid_3_2) == missing_val, NA,  as.numeric(data1()$mpid_3_2)),
    ifelse(as.numeric(data1()$mpid_3_5) == missing_val, NA, as.numeric(data1()$mpid_3_5)),
    ifelse(as.numeric(data1()$mpid_3_9) == missing_val, NA, as.numeric(data1()$mpid_3_9)),
    ifelse(as.numeric(data1()$mpid_3_11) == missing_val, NA, as.numeric(data1()$mpid_3_11)),
    as.numeric(data1()$mpid_3_15))), na.rm=TRUE),
    digits = 3)))
  
  data_in <- reactive(data.frame(
    PS = req(as.numeric(format(mean(as.numeric(c(
      data1()$mpid_1_1,
      data1()$mpid_1_7,
      data1()$mpid_1_12))),
      digits = 3))),
    
    I = req(as.numeric(format(mean(
        as.numeric(c(
          data1()$mpid_1_2,
          data1()$mpid_1_3,
          data1()$mpid_1_4,
          data1()$mpid_1_8,
          data1()$mpid_1_9,
          data1()$mpid_1_13,
          ifelse(as.numeric(data1()$mpid_1_14) == missing_val, NA, 
                 data1()$mpid_1_14),
          data1()$mpid_1_17,
          data1()$mpid_1_19,
          data1()$mpid_1_21)), 
        na.rm=TRUE),
      digits = 3))),
    
    LC = req(as.numeric(format(mean(as.numeric(c(
      data1()$mpid_1_11,
      data1()$mpid_1_16,
      data1()$mpid_1_18))),
      digits = 3))),

    AD = req(as.numeric( format(mean(as.numeric(c(6 - as.numeric(
      data1()$mpid_1_6),
      data1()$mpid_1_20,
      data1()$mpid_1_22))),
      digits = 3))),

    S = req(as.numeric(format(mean(as.numeric(c(
      data1()$mpid_1_5,
      data1()$mpid_1_10,
      data1()$mpid_1_15))),
      digits = 3))),

    PR = req(as.numeric(format(mean(as.numeric(c(
      data1()$mpid_2_2,
      data1()$mpid_2_5,
      data1()$mpid_2_8))),
      digits = 3))),

    SR = req(as.numeric(format(mean(as.numeric(c(
      data1()$mpid_2_1,
      data1()$mpid_2_3,
      data1()$mpid_2_6,
      data1()$mpid_2_10,
      data1()$mpid_2_11))),
      digits = 3))),

    DR = req(as.numeric(format(mean(as.numeric(c(
      data1()$mpid_2_4,
      data1()$mpid_2_7,
      data1()$mpid_2_9))),
      digits = 3))),

    GA = req(as.numeric(format((mpi_o() + mpi_h() + mpi_f()) / 3.0)), digits=3),

    Age =req(as.numeric(data1()$alter)),

    Sex = req(as.numeric(data1()$geschlecht))
    
  ))
  
  
  output$selected_var <- renderText({
    c(data_in()$PS, data_in()$I, data_in()$LC, data_in()$AD,
      data_in()$S, data_in()$PR, data_in()$SR, data_in()$DR, 
      data_in()$GA, data_in()$Age, data_in()$Sex) 
  })
 
  # You can access the value of the widget with input$action, e.g.
  output$value <- renderPrint({ input$save })
 
 predicted.classes <- reactive({predict(model, data_in(), type = "prob")})

 output$result <- renderText(
      paste0("Group probabilities: \n
             COPERS: ", round(predicted.classes()[3], digits = 2), ", ",
            "DYSFUNCTIONAL: ",round(predicted.classes()[1], digits = 2),", ",
            "DISTRESSED: ",round(predicted.classes()[2], digits = 2)))
 
 output$distPlot <- renderPlot({
   
   inputs = ((c(data_in()$PS, data_in()$I, data_in()$LC, data_in()$AD,
                data_in()$S, data_in()$PR, data_in()$SR, data_in()$DR, 
                data_in()$GA) - 
                model$all_mean[1:9])/model$all_sd[1:9]) * 10 + 50
   
   plot(inputs, xaxt = "n", type="b", pch=1, ylim=c(35, 70), lwd=c(6,4),
        ylab='T-SCORES', xlab="", cex=0.8, col="gray")
   mtext("PS = pain severity; I = interference;
     LC = life control; AD = affective distress; S = support;
     PR = punishing responses; SR = solicitous responses; DR = distracting
     responses; GA = general activity ", side=1, line=4.1, cex=0.8)
   mtext("Result", side=3)
   legend("topleft",
          c(paste("Prediction:", names[which.max(predicted.classes())]),
            paste("Dysfunctional", round(unlist(predicted.classes()[1]), 
                                         digits = 2)),
            paste("Distressed", round(unlist(predicted.classes()[2]), 
                                      digits = 2)),
            paste("Copers", round(unlist(predicted.classes()[3]), 
                                  digits = 2))),
          fill=c("gray", "red", "blue", "green"))
   
   points(model$cop[1:9], type="b", lty=2, lwd=2, pch=16, col="green")
   axis(1, at=1:9, labels=variables)
   points(model$dis[1:9], type="b", lty=2, lwd=2, pch=16, col="blue")
   points(model$dys[1:9], type="b", lty=2, lwd=2, pch=16, col="red")
 })
 
}

# Run the app ----
shinyApp(ui = ui, server = server)

