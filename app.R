# library(shiny)
packs = c("xgboost", "caret", "shiny")

if (length(setdiff(packs, rownames(installed.packages())))>0) {
  suppressMessages(install.packages(setdiff(packs, 
                                            rownames(installed.packages())),
                                    repos='http://cran.uni-muenster.de/'))
}
suppressMessages(invisible(lapply(packs, require, character.only=TRUE)))
suppressMessages(rm(packs))


#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
model <- readRDS("model.rds")
# Define UI ----
ui <- fluidPage (
  titlePanel("PerPain"),
  sidebarLayout(
     
    sidebarPanel(
      helpText("Select the input variables values:"),
      fluidRow(
      column(4, selectInput("Gender", h5("Sex:"),
                  choices = list("Male" = 1,
                                 "Female" = 2),
                  selected = 1)),
      column(4, numericInput("Age", h5("Age:"),
                  min = 0, max = 100, value = 50))),
      
      fluidRow(
         column(4,numericInput("PS", h5("Severity:"),
                               min = 0, max = 6, value = 3, step = 0.1)),
         column(4, numericInput("I", h5("Interference:"),
                  min = 0, max = 6, value = 3, step = 0.1)),
         
         column(4, numericInput("LC", h5("Life Control:"),
                  min = 0, max = 6, value = 3, step = 0.1))),
      fluidRow(
         column(4, numericInput("AD", h5("Distress:"),
                  min = 0, max = 6, value = 3, step = 0.1)),
         column(4, numericInput("S", h5("Support:"),
                  min = 0, max = 6, value = 3, step = 0.1)),
         column(4, numericInput("PR", h5("Punishing:"),
                  min = 0, max = 6, value = 3, step = 0.1))),
      fluidRow(
         column(4,numericInput("SR", h5(" Solicitous:"),
                  min = 0, max = 6, value = 3, step = 0.1)),
         column(4, numericInput("DR", h5("Distracting:"),
                  min = 0, max = 6, value = 3, step = 0.1)),
         column(4,numericInput("GA", h5("General:"),
                  min = 0, max = 6, value = 3, step = 0.1))),
      
      textOutput("result"),
    ),
    
    mainPanel(textOutput("Pred"),
              plotOutput(outputId = "distPlot"))
  )
)
names = c("COPERS", "DYSFUNCTIONAL", "DISTRESSED")
variables = c('PS', 'I', 'LC', 'AD', 'S', 'PR', 'SR', 'DR', 'GA')
# Define server logic ----
server <- function(input, output) {

 
 data <-reactive({
    if (input$Gender == "Female") {
       gender  = 1
    } else {
       gender  = 0
    }
   data.frame(
   PS = req(input$PS),
   I = req(input$I),
   LC = req(input$LC),
   AD =req(input$AD),
   S = req(input$S),
   PR =req(input$S),
   SR = req(input$SR),
   DR = req(input$DR),
   GA = req(input$GA),
   Age =req(input$Age),
   Gender = gender)
 })
 
 
 predicted.classes <- reactive({predict(model, data(), type = "prob")})
 # 
 output$result <- renderText(
      paste0("Group probabilities: \n
             COPERS: ", round(predicted.classes()[3], digits = 2), ", ",
            "DYSFUNCTIONAL: ",round(predicted.classes()[1], digits = 2),", ",
            "DISTRESSED: ",round(predicted.classes()[2], digits = 2)))
 
 output$distPlot <- renderPlot({
    axis(1, at=1:9, labels=variables)
    inputs = c(req(input$PS), input$I, input$LC, input$AD, input$S, input$PR, input$SR,
               input$DR, input$GA)
    plot(inputs, xaxt = "n", type="b", pch=1, ylim=c(0, 7), lwd=c(6,4),
         ylab='MEAN SCORES', xlab="", cex=0.8, col="gray")
    mtext("PS = pain severity; I = interference;
     LC = life control; AD = affective distress; S = support;
     PR = punishing responses; SR = solicitous responses; DR = distracting
     responses; GA = general activity ", side=1, line=4.1, cex=0.8)
    mtext("Result", side=3)
    legend("topleft",
           c(paste("Prediction:",names[which.max(predicted.classes())]),
             paste("Dysfunctional", round(unlist(predicted.classes()[1]), digits = 2)), 
             paste("Distressed", round(unlist(predicted.classes()[2]), digits = 2)), 
             paste("Copers", round(unlist(predicted.classes()[3]), digits = 2))),
           fill=c("gray", "red", "blue", "green"))
    axis(1, at=1:9, labels=variables)
    points(model$cop, type="b", lty=2, lwd=2, pch=16, col="green")
    points(model$dis, type="b", lty=2, lwd=2, pch=16, col="blue")
    points(model$dys, type="b", lty=2, lwd=2, pch=16, col="red")
 }, width = 1020, height = 480)
    
}

# Run the app ----
shinyApp(ui = ui, server = server)

