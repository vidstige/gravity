#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::cmp::max;

use eframe::{egui::{self, Ui, Sense, Id, PointerButton}, epaint::{Color32, Pos2, Vec2, Stroke, Rect}};

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 768.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Gravity App",
        options,
        Box::new(|_| { Box::<GravityApp>::default() }),
    )
}

struct Field {

}

impl Field {
    fn gradient(p: &Pos2) -> Vec2 {
        Vec2::new(0.0, 0.0)
    }
}

#[derive(PartialEq)]
enum Mode {
    Add,
    Remove,
}

struct GravityApp {
    field: Field,
    mode: Mode,
    radius: f32,
    orbital_velocity: bool,
}

impl Default for GravityApp {
    fn default() -> Self {
        Self {
            field: Field {  },
            mode: Mode::Add,
            radius: 64.0,
            orbital_velocity: true,
        }
    }
}

fn draw(ui: &mut Ui, field: &Field, spacing: Vec2) {
    let size = ui.available_size();
    let color = Color32::from_rgb(00, 0x8b, 0x8b);
    let stroke = Stroke { width: 1.0, color};
    let grid_size = size / spacing;
    let (grid_width, grid_height) = (grid_size.x as usize, grid_size.y as usize);
    let painter = ui.painter();
    let d10 = Vec2::new(spacing.x, 0.0);
    let d01 = Vec2::new(0.0, spacing.y);
    for y in 0..grid_height + 1 {
        for x in 0..grid_width + 1 {
            let p = Pos2::new(x as f32 * spacing.x, y as f32 * spacing.y);
            painter.line_segment([p, p + d10], stroke);
            painter.line_segment([p, p + d01], stroke);
        }
    }
}

impl eframe::App for GravityApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let id = Id::new("scroll_area");
        let spacing = Vec2::new(32.0, 32.0);
        egui::SidePanel::right("control_panel").show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.selectable_value(&mut self.mode, Mode::Add, "Add");
                ui.selectable_value(&mut self.mode, Mode::Remove, "Remove");
            });
            if self.mode == Mode::Add {
                ui.checkbox(&mut self.orbital_velocity, "orbital velocity");
            }
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            draw(ui, &self.field, spacing);
            let rect = Rect::from_min_size(Pos2::ZERO, ui.available_size());
            
            ui.input(|i| self.radius = (self.radius + 0.1 * i.scroll_delta.y).clamp(0.0, 1024.0));

            let response = ui.interact(rect, id, Sense::hover());
            if let Some(hover_pos) = response.hover_pos() {
                let stroke = Stroke { width: 1.0, color: Color32::WHITE};
                ui.painter().circle(hover_pos, self.radius, Color32::TRANSPARENT, stroke);
            }
            
            let response = ui.interact(rect, id, Sense::click());
            if response.clicked() {
                if let Some(pointer_pos) = response.interact_pointer_pos() {
                    println!("{:?}", pointer_pos);
                }
            }
        });
    }
}