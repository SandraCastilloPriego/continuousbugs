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
package GopiBugs.modules.configuration.tables.GCGC;

import GopiBugs.data.ParameterSet;
import GopiBugs.data.datamodels.DatasetGCGCDataModel;
import GopiBugs.data.impl.SimpleParameterSet;
import GopiBugs.desktop.Desktop;
import GopiBugs.desktop.GopiBugsMenu;
import GopiBugs.desktop.impl.DesktopParameters;
import GopiBugs.main.GopiBugsCore;
import GopiBugs.main.GopiBugsModule;
import GopiBugs.taskcontrol.Task;
import GopiBugs.taskcontrol.TaskListener;
import GopiBugs.taskcontrol.TaskStatus;
import GopiBugs.util.dialogs.ExitCode;
import GopiBugs.util.dialogs.ParameterSetupDialog;
import GopiBugs.util.internalframe.DataInternalFrame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.util.logging.Logger;
import javax.swing.JInternalFrame;
import javax.swing.JTable;
import javax.swing.table.TableModel;

/**
 *
 * @author scsandra
 */
public class GCGCColumnsView implements GopiBugsModule, TaskListener, ActionListener {

    private Logger logger = Logger.getLogger(this.getClass().getName());
    private Desktop desktop;
    private SimpleParameterSet parameters;

    public void initModule() {
        this.desktop = GopiBugsCore.getDesktop();
        desktop.addMenuItem(GopiBugsMenu.CONFIGURATION, "GCGC Table View..",
                "TODO write description", KeyEvent.VK_G, this, null, "icons/conf2.png");
        parameters = new GCGCColumnsViewParameters();

    }

    public void taskStarted(Task task) {
        logger.info("Running GCGC Table View");
    }

    public void taskFinished(Task task) {
        if (task.getStatus() == TaskStatus.FINISHED) {
            logger.info("Finished GCGC Table View ");
        }

        if (task.getStatus() == TaskStatus.ERROR) {

            String msg = "Error while GCGC Table View  .. ";
            logger.severe(msg);
            desktop.displayErrorMessage(msg);

        }
    }

    public void actionPerformed(ActionEvent e) {

        ExitCode exitCode = setupParameters();
        if (exitCode != ExitCode.OK) {
            return;
        }
        ((DesktopParameters) desktop.getParameterSet()).setViewGCGCParameters((GCGCColumnsViewParameters) parameters);
        runModule();
    }

    public ExitCode setupParameters() {
        try {
            ParameterSetupDialog dialog = new ParameterSetupDialog("GCGC Table View parameters", parameters);
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
        parameters = (GCGCColumnsViewParameters) parameterValues;
    }

    @Override
    public String toString() {
        return "LCMS Table View";
    }

    public Task[] runModule() {
        JInternalFrame[] frames = desktop.getInternalFrames();
        for (int i = 0; i < frames.length; i++) {
            try {
                JTable table = ((DataInternalFrame) frames[i]).getTable();
                TableModel model = table.getModel();
                if (model.getClass().toString().contains("DatasetGCGCDataModel")) {
                    ((DatasetGCGCDataModel) model).setParameters();
                }
                table.setModel(model);
                table.createDefaultColumnsFromModel();
                table.revalidate();
            } catch (Exception e) {
            }
        }
        return null;
    }
}
