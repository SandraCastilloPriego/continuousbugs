/*
 * Copyright 2007-2010 VTT Biotechnology
 * This file is part of GopiBugs.
 *
 * GopiBugs is free software; you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 *
 * GopiBugs is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * GopiBugs; if not, write to the Free Software Foundation, Inc., 51 Franklin St,
 * Fifth Floor, Boston, MA 02110-1301 USA
 */
package GopiBugs.modules.simulation.test;

import GopiBugs.data.BugDataset;
import GopiBugs.data.PeakListRow;
import GopiBugs.data.impl.SimpleParameterSet;
import GopiBugs.taskcontrol.Task;
import GopiBugs.taskcontrol.TaskStatus;
import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 *
 * @author scsandra
 */
public class TestTask implements Task {

        BugDataset training, validation;
        private Classifier classifier;
        double spec = 0, sen = 0, totalspec = 0, totalsen = 0;
        List<Integer> ids;
        Instances trainingData, validationData;
        double ridge = 0;
        private TaskStatus status = TaskStatus.WAITING;
        private String errorMessage;

        public TestTask(BugDataset training, BugDataset validation, SimpleParameterSet parameters) {
                this.training = training;
                if (validation == null) {
                        this.validation = training;
                } else {
                        this.validation = validation;
                }

                ridge = (Double) parameters.getParameterValue(TestParameters.ridge);
                String id = (String) parameters.getParameterValue(TestParameters.ids);
                String[] idsS = id.split(",");
                ids = new ArrayList<Integer>();
                for (String idS : idsS) {
                        ids.add(Integer.valueOf(idS));
                }

        }

        public String getTaskDescription() {
                return "Test concrete rows... ";
        }

        public double getFinishedPercentage() {
                return 0.0;
        }

        public TaskStatus getStatus() {
                return status;
        }

        public String getErrorMessage() {
                return errorMessage;
        }

        public void cancel() {
                status = TaskStatus.CANCELED;
        }

        public void run() {
                try {
                        status = TaskStatus.PROCESSING;

                        this.classify();
                        this.printPrediction();


                        status = TaskStatus.FINISHED;
                } catch (Exception e) {
                        status = TaskStatus.ERROR;
                }
        }

        public double prediction() {
                try {
                        Evaluation eval = new Evaluation(trainingData);
                        eval.evaluateModel(classifier, trainingData);
                        for (int i = 0; i < trainingData.numInstances(); i++) {
                                System.out.println(trainingData.instance(i).classValue() + " - " + eval.evaluateModelOnce(classifier, trainingData.instance(i)));
                        }

                        return eval.errorRate();

                } catch (Exception ex) {
                        ex.printStackTrace();
                        return 0.0;
                }
        }

        public void printPrediction() {
                try {
                        validationData = this.getDataset(validation);
                        Evaluation eval = new Evaluation(validationData);
                        eval.evaluateModel(classifier, validationData);
                        for (int i = 0; i < validationData.numInstances(); i++) {
                                System.out.println(validationData.instance(i).classValue() + " - " + eval.evaluateModelOnce(classifier, validationData.instance(i)));
                        }

                        System.out.println(eval.toSummaryString());
                        System.out.println(classifier.toString());
                } catch (Exception ex) {
                        ex.printStackTrace();
                }
        }

        private void classify() {
                try {
                        trainingData = this.getDataset(this.training);
                        classifier = new LinearRegression();
                        ((LinearRegression) classifier).setRidge(this.ridge);
                        classifier.buildClassifier(trainingData);
                } catch (Exception ex) {
                }
        }

        private Instances getDataset(BugDataset dataset) {
                FastVector attributes = new FastVector();

                for (int i = 0; i < ids.size(); i++) {
                        Attribute weight = new Attribute("weight" + i);
                        attributes.addElement(weight);
                }


                Attribute type = new Attribute("class");

                attributes.addElement(type);

                //Creates the dataset
                Instances train = new Instances("Dataset", attributes, 0);

                for (int i = 0; i < dataset.getNumberCols(); i++) {
                        double[] values = new double[train.numAttributes()];
                        String sampleName = dataset.getAllColumnNames().elementAt(i);
                        int cont = 0;
                        for (Integer id : ids) {
                                for (PeakListRow row : dataset.getRows()) {
                                        if (row.getID() == id) {
                                                values[cont++] = (Double) row.getPeak(sampleName);
                                        }
                                }
                        }
                        values[cont] = Double.parseDouble(dataset.getSampleType(sampleName));

                        Instance inst = new SparseInstance(1.0, values);
                        train.add(inst);
                }

                train.setClass(type);

                return train;

        }
}
