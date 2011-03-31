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

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import javax.swing.JDesktopPane;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import ca.guydavis.swing.desktop.CascadingWindowPositioner;
import ca.guydavis.swing.desktop.JWindowsMenu;
import GopiBugs.desktop.GopiBugsMenu;
import GopiBugs.main.GopiBugsCore;
import javax.swing.ImageIcon;

/**
 * @author Taken from MZmine2
 * http://mzmine.sourceforge.net/
 */
class MainMenu extends JMenuBar implements ActionListener {

    private JMenu fileMenu, configurationMenu, controlMenu, helpMenu;
    private JWindowsMenu windowsMenu;
    private JMenuItem hlpAbout;

    MainMenu() {

        fileMenu = new JMenu("File");
        fileMenu.setMnemonic(KeyEvent.VK_F);
        add(fileMenu);

        configurationMenu = new JMenu("Configuration");
        configurationMenu.setMnemonic(KeyEvent.VK_C);
        add(configurationMenu);

        controlMenu = new JMenu("Control");
        controlMenu.setMnemonic(KeyEvent.VK_C);
        add(controlMenu);

        JDesktopPane mainDesktopPane = ((MainWindow) GopiBugsCore.getDesktop()).getDesktopPane();
        windowsMenu = new JWindowsMenu(mainDesktopPane);
        CascadingWindowPositioner positioner = new CascadingWindowPositioner(
                mainDesktopPane);
        windowsMenu.setWindowPositioner(positioner);
        windowsMenu.setMnemonic(KeyEvent.VK_W);
        this.add(windowsMenu);

        helpMenu = new JMenu("Help");
        helpMenu.setMnemonic(KeyEvent.VK_H);
        this.add(helpMenu);

        hlpAbout = addMenuItem(GopiBugsMenu.HELPSYSTEM, "About GopiBugs..",
                "About GopiBugs..", KeyEvent.VK_A, this,
                null, null);
    }

    public void addMenuItem(GopiBugsMenu parentMenu, JMenuItem newItem) {
        switch (parentMenu) {
            case FILE:
                fileMenu.add(newItem);
                break;
            case CONFIGURATION:
                configurationMenu.add(newItem);
                break;
            case CONTROL:
                controlMenu.add(newItem);
                break;
            case HELPSYSTEM:
                helpMenu.add(newItem);
                break;
        }
    }

    public JMenuItem addMenuItem(GopiBugsMenu parentMenu, String text,
            String toolTip, int mnemonic,
            ActionListener listener, String actionCommand, String icon) {

        JMenuItem newItem = new JMenuItem(text);
        if (listener != null) {
            newItem.addActionListener(listener);
        }
        if (actionCommand != null) {
            newItem.setActionCommand(actionCommand);
        }
        if (toolTip != null) {
            newItem.setToolTipText(toolTip);
        }
        if (mnemonic > 0) {
            newItem.setMnemonic(mnemonic);
        }

        if (icon != null) {
            newItem.setIcon(new ImageIcon(icon));
        }
        addMenuItem(parentMenu, newItem);
        return newItem;

    }

    public void addMenuSeparator(GopiBugsMenu parentMenu) {
        switch (parentMenu) {
            case FILE:
                fileMenu.addSeparator();
                break;
            case CONFIGURATION:
                configurationMenu.addSeparator();
                break;
            case CONTROL:
                controlMenu.addSeparator();
                break;
            case HELPSYSTEM:
                helpMenu.addSeparator();
                break;

        }
    }

    /**
     * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
     */
    public void actionPerformed(ActionEvent e) {
        Object src = e.getSource();
        if (src == hlpAbout) {
            MainWindow mainWindow = (MainWindow) GopiBugsCore.getDesktop();
            mainWindow.showAboutDialog();
        }
    }
}
