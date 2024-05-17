# Case - Incognia

## The case

The instructions and the problems to solve can be found [here](https://drive.google.com/file/d/1KovNp4PxnbQ69FRNYWv-KS9xsRMYPyPA/view?usp=drive_link)

## Files

There is one notebook:

1. **case - incognia**: Contains the code to create the tables and perform the creation, analysis and comparison of the new decision flow vs current one. The language used to generate this analysis is SQL, Python and PySpark.

## Code

### Environment

The configs from the cluster that ran the notebooks are:

    Databricks Runtime Version
    > 14.3 LTS (includes Apache Spark 3.5.0, Scala 2.12)

    Driver type
    > Community Optimized
    > 15.3 GB Memory, 2 Cores, 1 DBU

### Depedencies

We didn't install any external libraries or dependencies besides the ones already present in the Databricks Community default configs.

There are a few steps to follow:

1. Make sure you have a Databricks Community account.
2. Create a cluster with the configs shown in the **Environment** section. You can create in the menu **Compute** -> **Create compute**.
3. Make sure the **default** database exists on **Catalog** -> **Databases**.
4. Load the files([Base 1 - Hélio Assakura - incognia-case-transactions-fraud-feedback.csv](https://drive.google.com/file/d/1L_vljs5bHtl-wjr2905gbUowgNml2a0i/view?usp=sharing), [Base 2 - Hélio Assakura - incognia-case-transactions.csv](https://drive.google.com/file/d/1MPYGHFtLIfcr4yeN11I49iJBpOa9ySYd/view?usp=sharing)) in Databricks. To do so, you can use the path **Catalog** -> **Create Table** -> **Drop Files To Upload**.


### Images

There's a folder named `images` that contains some graphs for the analysis. They are in the report, but you can use them to have a better image resolution.

## Instructions

### Importing the code

With the `case - incognia.py` downloaded, you can use the path **Workspace** -> **youruser@email** -> **Right click** -> **Import**.

### Running the Notebook

Assuming you have Databricks setup done and the source file imported, you can run the Notebook.

You should be able to run each cell in sequence. The output will be some of the analysis and results presented in the final case.