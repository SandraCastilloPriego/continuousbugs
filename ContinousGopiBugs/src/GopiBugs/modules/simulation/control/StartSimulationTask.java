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
package GopiBugs.modules.simulation.control;

import GopiBugs.data.BugDataset;
import GopiBugs.data.DatasetType;
import GopiBugs.data.PeakListRow;
import GopiBugs.data.impl.SimpleParameterSet;
import GopiBugs.main.GopiBugsCore;
import GopiBugs.modules.simulation.Bug;
import GopiBugs.modules.simulation.CanvasWorld;
import GopiBugs.modules.simulation.Result;
import GopiBugs.modules.simulation.TestBug;
import GopiBugs.modules.simulation.World;
import GopiBugs.modules.simulation.classifiersEnum;
import GopiBugs.taskcontrol.TaskStatus;
import GopiBugs.util.Range;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import javax.swing.JInternalFrame;
import javax.swing.JPanel;
import java.util.List;
import java.util.Random;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

/**
 *
 * @author scsandra
 */
public class StartSimulationTask {

        private BugDataset training, validation;
        private TaskStatus status = TaskStatus.WAITING;
        private String errorMessage;
        private sinkThread thread;
        private CanvasWorld canvas;
        private JPanel canvasPanel;
        private World world;
        private JInternalFrame frame;
        private int numberOfBugsCopies, worldSize, bugLife, iterations, maxBugs = 1000;
        private JTextArea textArea;
        private List<Range> ranges;
        private Random rand;
        private int totalIDs, stoppingCriteria, stopCounting = 0;
        private List<Result> results;
        private double minCountId = 100;
        private double maxScoreForReproduction;

        public StartSimulationTask(BugDataset[] datasets, SimpleParameterSet parameters) {
                for (BugDataset dataset : datasets) {
                        if (dataset.getType() == DatasetType.TRAINING) {
                                training = dataset;
                                this.totalIDs = training.getNumberRows();
                        } else if (dataset.getType() == DatasetType.VALIDATION) {
                                validation = dataset;
                        }
                }
                this.numberOfBugsCopies = (Integer) parameters.getParameterValue(StartSimulationParameters.numberOfBugs);
                this.worldSize = (Integer) parameters.getParameterValue(StartSimulationParameters.worldSize);
                this.bugLife = (Integer) parameters.getParameterValue(StartSimulationParameters.bugLife);
                this.iterations = (Integer) parameters.getParameterValue(StartSimulationParameters.iterations);
                this.maxBugs = (Integer) parameters.getParameterValue(StartSimulationParameters.bugLimit);
                this.stoppingCriteria = (Integer) parameters.getParameterValue(StartSimulationParameters.stoppingCriteria);
                this.maxScoreForReproduction = (Double) parameters.getParameterValue(StartSimulationParameters.maxScore);

                this.ranges = new ArrayList<Range>();
                this.results = new ArrayList<Result>();
                this.rand = new Random();
        }

        public String getTaskDescription() {
                return "Start simulation... ";
        }

        public double getFinishedPercentage() {
                return 0.0f;
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
                        frame = new JInternalFrame("Simulation", true, true, true, true);
                        frame.setSize(new Dimension(700, 700));
                        frame.setLayout(new FlowLayout());
                        canvasPanel = new JPanel();
                        canvasPanel.setLayout(new FlowLayout());
                        canvasPanel.setSize(new Dimension(700, 700));
                        canvasPanel.setVisible(true);
                        frame.add(canvasPanel);
                        GopiBugsCore.getDesktop().addInternalFrame(frame);


                        JInternalFrame frame2 = new JInternalFrame("Results", true, true, true, true);
                        frame2.setSize(new Dimension(700, 700));

                        textArea = new JTextArea("");
                        textArea.setSize(new Dimension(700, 700));
                        JScrollPane panel = new JScrollPane(textArea);
                        frame2.add(panel);
                        GopiBugsCore.getDesktop().addInternalFrame(frame2);
                        createRanges();
                        int index = rand.nextInt(ranges.size() - 1);
                        Range range = ranges.get(index);
                        this.startCicle(range, null, null);
                        status = TaskStatus.FINISHED;
                } catch (Exception e) {
                        e.printStackTrace();
                        status = TaskStatus.ERROR;
                }
        }

        private void startCicle(Range range, List<Bug> bugs, List<Result> results) {
                world = new World(training, validation, this.worldSize, range, bugs, this.numberOfBugsCopies, this.bugLife, textArea, results, this.maxBugs, this.maxScoreForReproduction);
                canvas = new CanvasWorld(world);
                canvasPanel.removeAll();
                canvasPanel.add(canvas);
                canvas.setVisible(true);
                if (thread != null) {
                        thread.interrupt();
                }
                thread = new sinkThread();
                thread.start();
        }

        private void createRanges() {
                int cont = 0;
                int unit = training.getNumberCols() / 10;
                while (cont < 11) {
                        this.ranges.add(new Range(unit * cont, (unit * cont) + unit));
                        cont++;
                }

        }

        private double countIDs() {
                int count = 0;
                List<Integer> alreadyCount = new ArrayList<Integer>();
                for (Bug bug : world.getBugs()) {
                        for (PeakListRow row : bug.getRows()) {
                                if (!alreadyCount.contains(row.getID())) {
                                        alreadyCount.add(row.getID());
                                        count++;
                                }

                        }
                }
                double result = ((double) ((double) count / (double) this.totalIDs)) * 100;
                System.out.println("Count : " + count + "/" + this.totalIDs + " result: " + result + "%");
                return result;
        }

        public class sinkThread extends Thread {

                @Override
                public void run() {
                        for (int i = 0; i < iterations; i++) {
                                // Paint the graphics
                                canvas.update(canvas.getGraphics());
                                world.cicle();
                        }
                        int index = rand.nextInt(ranges.size() - 1);
                        Range range = ranges.get(index);
                        printResult(world.getBugs());
                        double result = countIDs();



                        if (result < minCountId) {
                                minCountId = result;
                                stopCounting = 0;
                        } else {
                                stopCounting++;
                        }
                        // Checking the stopping criteria
                        if (result < stoppingCriteria || stopCounting > 5000) {
                                Result res = results.get(0);
                                List<Integer> ids = new ArrayList<Integer>();
                                for (String values : res.getValues()) {
                                        ids.add(Integer.valueOf(values));
                                }
                                TestBug test = new TestBug(ids, classifiersEnum.LinearRegression, training, validation, res.ridge);
                                test.printPrediction();

                        } else {
                                startCicle(range, world.getBugs(), world.getResult());
                        }

                }
        }

        public void printResult(List<Bug> bugs) {

                Comparator<Result> c = new Comparator<Result>() {

                        public int compare(Result o1, Result o2) {
                                if (o1.count < o2.count) {
                                        return 1;
                                } else {
                                        return -1;
                                }
                        }
                };

                for (int i = 0; i < 30; i++) {
                        Bug bug = bugs.get(i);
                        if (bug.getScore() < this.maxScoreForReproduction) {
                                Result result = new Result();
                                result.Classifier = bug.getClassifierType().name();
                                List<Integer> ids = new ArrayList<Integer>();
                                for (PeakListRow row : bug.getRows()) {
                                        result.addValue(String.valueOf(row.getID()));
                                        ids.add(row.getID());
                                }

                                TestBug testing = new TestBug(ids, bug.getClassifierType(), training, validation, bug.getRidge());
                                double value = testing.prediction();
                                result.realscore = value;
                                result.score = bug.getScore();
                                result.ridge = bug.getRidge();
                                boolean isIt = false;
                                for (Result r : this.results) {
                                        if (r.isIt(result.getValues(), result.Classifier)) {
                                                r.count();
                                                isIt = true;
                                        }
                                }

                                if (!isIt) {
                                        this.results.add(result);
                                }
                        }
                }


                Collections.sort(results, c);

                int contbug = 0;
                String result = "";

                for (Result r : results) {
                        result += r.toString();
                        contbug++;
                        if (contbug > 30) {
                                break;
                        }
                }

                this.textArea.setText(result);
                //  System.out.println(result);

        }
}
