packs = c("shiny")

if (length(setdiff(packs, rownames(installed.packages())))>0) {
  suppressMessages(install.packages(setdiff(packs, 
                                            rownames(installed.packages())),
                                    repos='http://cran.uni-muenster.de/'))
}
suppressMessages(invisible(lapply(packs, require, character.only=TRUE)))
suppressMessages(rm(packs))

folder_address = getwd()
runApp(folder_address, launch.browser=TRUE)