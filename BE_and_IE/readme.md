## Batch Enrichment (BE) and Incremental Enrichment (IE)

  

## 1. Installation
Before building the projects, the following prerequisites need to be installed:
* Java JDK 1.8
* Maven

## 2. mls-server.zip    
The compressed file is the source code for BE and IE.

1. Put the datasets into target directory:

2. Compile and build the project:
```
mvn package
```
3. Move and replace the **mls-server-0.1.1.jar** from mls-server/target/ to a lib file:

```
mv target/mls_server-0.1.1.jar example/lib/
```
## 3. datasets_and_shell
The directory contains IMDB and Person datasets, and shell scripts to run the experimental settings.

## 4. dataset file

Due to the large size of dataset for BE and IE, please download the Zip file from [link](https://www.dropbox.com/sh/kk84zjrgwa0dikc/AABZ6MueJGc03MsZIbw8k-cra?dl=0) , then replace the BE_and_IE folder with the downloaded version.
