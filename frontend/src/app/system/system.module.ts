import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { SystemRoutingModule } from './system-routing.module';
import { SystemComponent } from './system.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { CommonModule } from "@angular/common";
import { AuthModule } from '../auth/auth.module';


@NgModule({
  declarations: [
    SystemComponent
  ],
  imports: [
    SystemRoutingModule,
    ReactiveFormsModule,
    HttpClientModule,
    CommonModule,
    FormsModule,
    BrowserModule,
    AuthModule
  ],
  bootstrap: [SystemComponent]
})
export class SystemModule { }
