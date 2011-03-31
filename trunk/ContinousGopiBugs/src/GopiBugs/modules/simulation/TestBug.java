/*
 * Copyright 2007-2010 
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
package GopiBugs.modules.simulation;

import GopiBugs.data.BugDataset;
import GopiBugs.data.PeakListRow;
import java.util.List;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.ComplementNaiveBayes;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.KStar;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.AdditiveRegression;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.MultiScheme;
import weka.classifiers.meta.RandomCommittee;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.meta.Stacking;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.lmt.LogisticBase;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 *
 *
 * @author SCSANDRA
 */
public class TestBug {

        BugDataset training, validation;
        private Classifier classifier;
        double spec = 0, sen = 0, totalspec = 0, totalsen = 0;
        List<Integer> ids;
        classifiersEnum classifierType;
        Instances data;
        double ridge = 0;

        public TestBug(List<Integer> ids, classifiersEnum classifier, BugDataset training, BugDataset validation, double ridge) {
                this.training = training;
                this.validation = validation;
                this.ids = ids;
                classifierType = classifier;
                this.classify();
                this.prediction();
                this.ridge = ridge;
        }

        public void addId(int id) {
                this.ids.add(id);
        }

        private void classify() {
                try {
                        data = this.getDataset(this.training);
                        classifier = new LinearRegression();
                        ((LinearRegression) classifier).setRidge(this.ridge);
                        // this.classifier = new AdditiveRegression();
                        classifier.buildClassifier(data);
                } catch (Exception ex) {
                }
        }

        public double prediction() {
                try {


                        Evaluation eval = new Evaluation(data);
                        eval.evaluateModel(classifier, data);
                        /* for(int i = 0; i < data.numInstances(); i++){
                        System.out.println(data.instance(i).classValue() +" - " + eval.evaluateModelOnce(classifier, data.instance(i)));
                        }*/

                        return eval.errorRate();

                } catch (Exception ex) {
                        ex.printStackTrace();
                        return 0.0;
                }
        }

        public void printPrediction() {
                try {
                        Evaluation eval = new Evaluation(data);
                        eval.evaluateModel(classifier, data);
                        for (int i = 0; i < data.numInstances(); i++) {
                                System.out.println(data.instance(i).classValue() + " - " + eval.evaluateModelOnce(classifier, data.instance(i)));
                        }
                } catch (Exception ex) {
                        ex.printStackTrace();
                }
        }


        /* public double isClassify(String sampleName) {
        FastVector attributes = new FastVector();
        for (int i = 0; i < this.ids.size(); i++) {
        Attribute weight = new Attribute("weight" + i);
        attributes.addElement(weight);
        }

        Attribute type = new Attribute("class");
        attributes.addElement(type);
        //Creates the dataset
        Instances cellDataset = new Instances("Train Dataset", attributes, 0);
        double[] values = new double[cellDataset.numAttributes()];
        int cont = 0;
        for (PeakListRow row : rowList) {
        values[cont++] = (Double) row.getPeak(sampleName);
        }
        values[cont] = Double.parseDouble(this.dataset.getSampleType(sampleName));
        Instance inst = new SparseInstance(1.0, values);
        cellDataset.add(inst);
        cellDataset.setClassIndex(cont);
        try {
        return classifier.classifyInstance(cellDataset.instance(0));
        } catch (Exception ex) {
        }
        return 1000;
        }*/
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

        private Classifier getClassifier(classifiersEnum classifierType) {
                switch (classifierType) {
                        case Logistic:
                                return new Logistic();
                        case LogisticBase:
                                return new LogisticBase();
                        case LogitBoost:
                                return new LogitBoost();
                        case NaiveBayesMultinomialUpdateable:
                                return new NaiveBayesMultinomialUpdateable();
                        case NaiveBayesUpdateable:
                                return new NaiveBayesUpdateable();
                        case RandomForest:
                                return new RandomForest();
                        case RandomCommittee:
                                return new RandomCommittee();
                        case RandomTree:
                                return new RandomTree();
                        case ZeroR:
                                return new ZeroR();
                        case Stacking:
                                return new Stacking();
                        case AdaBoostM1:
                                return new AdaBoostM1();
                        case Bagging:
                                return new Bagging();
                        case ComplementNaiveBayes:
                                return new ComplementNaiveBayes();
                        case IB1:
                                return new IB1();
                        case J48:
                                return new J48();
                        case KStar:
                                return new KStar();
                        case LMT:
                                return new LMT();
                        case MultiScheme:
                                return new MultiScheme();
                        case NaiveBayes:
                                return new NaiveBayes();
                        case NaiveBayesMultinomial:
                                return new NaiveBayesMultinomial();
                        case OneR:
                                return new OneR();
                        case PART:
                                return new PART();
                        case RandomSubSpace:
                                return new RandomSubSpace();
                        case REPTree:
                                return new REPTree();
                        case SimpleLogistic:
                                return new SimpleLogistic();
                        case SMO:
                                return new SMO();
                        case LinearRegression:
                                return new LinearRegression();
                        default:
                                return null;
                }
        }
}
