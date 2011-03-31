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

import GopiBugs.modules.file.saveGCGCFile.*;
import GopiBugs.data.BugDataset;
import GopiBugs.data.ParameterSet;
import GopiBugs.data.impl.SimpleParameterSet;
import GopiBugs.desktop.Desktop;
import GopiBugs.desktop.GopiBugsMenu;
import GopiBugs.main.GopiBugsCore;
import GopiBugs.main.GopiBugsModule;
import GopiBugs.taskcontrol.Task;
import GopiBugs.taskcontrol.TaskListener;
import GopiBugs.taskcontrol.TaskStatus;
import GopiBugs.util.dialogs.ExitCode;
import GopiBugs.util.dialogs.ParameterSetupDialog;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.util.logging.Logger;

/**
 *
 * @author scsandra
 */
public class Test implements GopiBugsModule, TaskListener, ActionListener{

    private Logger logger = Logger.getLogger(this.getClass().getName());
    private Desktop desktop;
    private SimpleParameterSet parameters;   

    public void initModule() {
        this.desktop = GopiBugsCore.getDesktop();
            desktop.addMenuItem(GopiBugsMenu.CONTROL, "Test concrete rows..",
                    "Test concrete rows", KeyEvent.VK_L, this, null, null);
            parameters = new TestParameters();
    }

    public void taskStarted(Task task) {
        logger.info("Running Test concrete rows");
    }

     public void actionPerformed(ActionEvent e) {
                ExitCode exitCode = setupParameters();
                if (exitCode != ExitCode.OK) {
                        return;
                }

                runModule();
        }

    public void taskFinished(Task task) {
        if (task.getStatus() == TaskStatus.FINISHED) {
            logger.info("Finished Test concrete rows" + ((SaveGCGCFileTask) task).getTaskDescription());
        }

        if (task.getStatus() == TaskStatus.ERROR) {

            String msg = "Error while Test concrete rows on .. " + ((SaveGCGCFileTask) task).getErrorMessage();
            logger.severe(msg);
            desktop.displayErrorMessage(msg);

        }
    }

    public ExitCode setupParameters() {
        try {
            ParameterSetupDialog dialog = new ParameterSetupDialog("parameters", parameters);
            dialog.setVisible(true);
            return dialog.getExitCode();
        } catch (Exception exception) {
            return ExitCode.CANCEL;
        }
    }

    public ParameterSet getParameterSet() {
        return parameters;
    }

    public void setParameters(ParameterSet parameterValues) {
        parameters = (SimpleParameterSet) parameterValues;
    }

    @Override
    public String toString() {
        return "Test";
    }

    public Task[] runModule() {

        // prepare a new group of tasks
        BugDataset[] datasets = desktop.getSelectedDataFiles();
        Task tasks[] = new TestTask[datasets.length];
        for (int i = 0; i < datasets.length; i++) {
            tasks[i] = new TestTask(datasets[i], parameters);
        }

        GopiBugsCore.getTaskController().addTasks(tasks);

        return tasks;

    }
}
