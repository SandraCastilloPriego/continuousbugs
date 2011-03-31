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
package GopiBugs.desktop.impl;

import GopiBugs.data.BugDataset;
import GopiBugs.data.ParameterSet;
import GopiBugs.desktop.Desktop;
import GopiBugs.desktop.GopiBugsMenu;
import GopiBugs.desktop.impl.helpsystem.GopiBugsHelpSet;
import GopiBugs.main.GopiBugsCore;
import GopiBugs.main.GopiBugsModule;
import GopiBugs.taskcontrol.impl.TaskProgressWindow;
import GopiBugs.util.ExceptionUtils;
import GopiBugs.util.TextUtils;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;

import javax.help.HelpBroker;
import javax.swing.JDesktopPane;
import javax.swing.JFrame;
import javax.swing.JInternalFrame;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JSplitPane;
import javax.swing.border.EtchedBorder;

/**
 * This class is the main window of application
 *
 * @author Taken from MZmine2
 * http://mzmine.sourceforge.net/
 * 
 */
public class MainWindow extends JFrame implements GopiBugsModule, Desktop,
        WindowListener {

    static final String aboutHelpID = "GopiBugs/desktop/help/AboutGopiBugs.html";
    private DesktopParameters parameters;
    private JDesktopPane desktopPane;
    private JSplitPane split;
    private ItemSelector itemSelector;
    private TaskProgressWindow taskList;

    public TaskProgressWindow getTaskList() {
        return taskList;
    }
    private MainMenu menuBar;
    private Statusbar statusBar;

    public MainMenu getMainMenu() {
        return menuBar;
    }

    public void addInternalFrame(JInternalFrame frame) {
        try {
            desktopPane.add(frame);
            frame.setVisible(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // TODO: adjust frame position

    }

    /**
     * This method returns the desktop
     */
    public JDesktopPane getDesktopPane() {
        return desktopPane;
    }

    /**
     * WindowListener interface implementation
     */
    public void windowOpened(WindowEvent e) {
    }

    public void windowClosing(WindowEvent e) {
        GopiBugsCore.exitGopiBugs();
    }

    public void windowClosed(WindowEvent e) {
    }

    public void windowIconified(WindowEvent e) {
    }

    public void windowDeiconified(WindowEvent e) {
    }

    public void windowActivated(WindowEvent e) {
    }

    public void windowDeactivated(WindowEvent e) {
    }

    public void setStatusBarText(String text) {
        setStatusBarText(text, Color.black);
    }

    /**
     */
    public void displayMessage(String msg) {
        displayMessage("Message", msg, JOptionPane.INFORMATION_MESSAGE);
    }

    /**
     */
    public void displayMessage(String title, String msg) {
        displayMessage(title, msg, JOptionPane.INFORMATION_MESSAGE);
    }

    public void displayErrorMessage(String msg) {
        displayMessage("Error", msg);
    }

    public void displayErrorMessage(String title, String msg) {
        displayMessage(title, msg, JOptionPane.ERROR_MESSAGE);
    }

    public void displayMessage(String title, String msg, int type) {
        String wrappedMsg = TextUtils.wrapText(msg, 80);
        JOptionPane.showMessageDialog(this, wrappedMsg, title, type);
    }

    public void addMenuItem(GopiBugsMenu parentMenu, JMenuItem newItem) {
        menuBar.addMenuItem(parentMenu, newItem);
    }

    /**
     */
    public void initModule() {

        SwingParameters.initSwingParameters();

        parameters = new DesktopParameters();

        /*  try {
        BufferedImage GopiBugsIcon = ImageIO.read(new File(
        "icons/GopiBugsIcon.png"));
        setIconImage(GopiBugsIcon);
        } catch (IOException e) {
        e.printStackTrace();
        }*/

        // Initialize item selector
        itemSelector = new ItemSelector(this);

        // Place objects on main window
        desktopPane = new JDesktopPane();

        split = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT, itemSelector,
                desktopPane);

        desktopPane.setDragMode(JDesktopPane.OUTLINE_DRAG_MODE);

        desktopPane.setBorder(new EtchedBorder(EtchedBorder.RAISED));
        // desktopPane.setBackground(new Color(237, 249, 255));
        Container c = getContentPane();
        c.setLayout(new BorderLayout());
        c.add(split, BorderLayout.CENTER);

        statusBar = new Statusbar();
        c.add(statusBar, BorderLayout.SOUTH);

        // Construct menu
        menuBar = new MainMenu();
        setJMenuBar(menuBar);

        // Initialize window listener for responding to user events
        addWindowListener(this);

        pack();

        // TODO: check screen size?
        setBounds(0, 0, 1000, 700);
        setLocationRelativeTo(null);

        // Application wants to control closing by itself
        setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
        updateTitle();

        setTitle("GopiBugs");

//        taskList = new TaskProgressWindow();
        //  desktopPane.add(taskList, JLayeredPane.DEFAULT_LAYER);

    }

    void updateTitle() {
        setTitle("GopiBugs " + GopiBugsCore.getGopiBugsVersion());
    }

    public JFrame getMainFrame() {
        return this;
    }

    public JMenuItem addMenuItem(GopiBugsMenu parentMenu, String text,
            String toolTip, int mnemonic, ActionListener listener,
            String actionCommand, String icon) {
        return menuBar.addMenuItem(parentMenu, text, toolTip, mnemonic,
                listener, actionCommand, icon);
    }

    public void addMenuSeparator(GopiBugsMenu parentMenu) {
        menuBar.addMenuSeparator(parentMenu);

    }

    public JInternalFrame getSelectedFrame() {
        return desktopPane.getSelectedFrame();
    }

    public JInternalFrame[] getInternalFrames() {
        return desktopPane.getAllFrames();
    }

    public void setStatusBarText(String text, Color textColor) {
        statusBar.setStatusText(text, textColor);
    }

    public DesktopParameters getParameterSet() {
        return parameters;
    }

    public void setParameters(ParameterSet parameterValues) {
        this.parameters = (DesktopParameters) parameterValues;
    }

    public ItemSelector getItemSelector() {
        return itemSelector;
    }

    public BugDataset[] getSelectedDataFiles() {
        return this.itemSelector.getSelectedDatasets();
    }

    /*public Vector[] getSelectedExperiments() {
    return this.itemSelector.getSelectedExperiments();
    }*/
    public void AddNewFile(BugDataset dataset) {
        this.itemSelector.addNewFile(dataset);
    }

    public void removeData(BugDataset file) {
        this.itemSelector.removeData(file);
    }

    public void displayException(Exception e) {
        displayErrorMessage(ExceptionUtils.exceptionToString(e));
    }

    public void showAboutDialog() {
        GopiBugsHelpSet hs = GopiBugsCore.getHelpImpl().getHelpSet();
        if (hs == null) {
            return;
        }

        HelpBroker hb = hs.createHelpBroker();
        hs.setHomeID(aboutHelpID);

        hb.setDisplayed(true);
    }
}
