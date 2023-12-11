#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::{egui::{self, Ui}, epaint::{Color32, Pos2, Vec2, Stroke}};

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
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

struct GravityApp {
    field: Field,
}

impl Default for GravityApp {
    fn default() -> Self {
        Self {
            field: Field {  },
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
        let spacing = Vec2::new(32.0, 32.0);
        egui::CentralPanel::default().show(ctx, |ui| {
            draw(ui, &self.field, spacing);
            /*
            ui.heading("Gravity");ui.horizontal(|ui| {
                let name_label = ui.label("Your name: ");
                ui.text_edit_singleline(&mut self.name)
                    .labelled_by(name_label.id);
            });
            ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));
            if ui.button("Click each year").clicked() {
                self.age += 1;
            }
            ui.label(format!("Hello '{}', age {}", self.name, self.age));*/
           
        });
    }
}